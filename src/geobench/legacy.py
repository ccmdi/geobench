import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property
from math import isclose
from pathlib import Path
from typing import Any

from geobench.round import MAX_POINTS, Round
from geobench.taskset import GeoTasksetConfig
from shungite import Bounds, Coordinates, Guess, Item
from shungite.report import Median

FOLDER = re.compile(r"(?P<model>.+)_(?P<map>[^_]+)_(?P<started>\d{4}-\d\d-\d\dT\d\d_\d\d_\d\d)")
STARTED = "%Y-%m-%dT%H_%M_%S"
SAME_POINT = 1e-9
SAME_SUM = 1e-12


@dataclass(frozen=True)
class Row:
    id: str
    cells: dict[str, str]

    def number(self, column: str) -> float | None:
        cell = self.cells.get(column)
        return float(cell) if cell else None

    @property
    def round(self) -> dict[str, Any]:
        truth = Coordinates(lat=float(self.cells["lat_true"]), lng=float(self.cells["lng_true"]))
        return {"id": self.id, "country": self.cells["country_true"], "location": truth.model_dump()}

    @property
    def guess(self) -> Guess | None:
        lat, lng = self.number("lat_guess"), self.number("lng_guess")
        if lat is None or lng is None:
            return None
        return Guess(country=self.cells["country_guess"] or None, coordinates=Coordinates(lat=lat, lng=lng))

    @property
    def item(self) -> Item:
        points, km, guess = self.number("score"), self.number("distance_km"), self.guess
        return Item(
            score=0.0 if points is None else points / MAX_POINTS,
            passed=self.cells.get("country_correct") == "True",
            guess=None if guess is None else guess.model_dump(),
            refused=self.cells["refused"] == "True",
            **{name: value for name, value in (("points", points), ("distance_km", km)) if value is not None},
            **({"error_message": error} if (error := self.cells["error_message"]) else {}),
        )


def provider_usage(response: dict[str, Any]) -> dict[str, int] | None:
    if gemini := response.get("usageMetadata"):
        return {"prompt_tokens": gemini["promptTokenCount"], "completion_tokens": gemini.get("candidatesTokenCount", 0) + gemini.get("thoughtsTokenCount", 0)}
    spent = response.get("usage")
    if not spent:
        return None
    if "prompt_tokens" in spent:
        return {"prompt_tokens": spent["prompt_tokens"], "completion_tokens": spent["completion_tokens"]}
    cached = (spent.get("cache_read_input_tokens") or 0) + (spent.get("cache_creation_input_tokens") or 0)
    return {"prompt_tokens": spent["input_tokens"] + cached, "completion_tokens": spent["output_tokens"]}


@dataclass(frozen=True)
class LegacyRun:
    dir: Path
    dataset: Path

    @cached_property
    def folder(self) -> re.Match[str]:
        named = FOLDER.fullmatch(self.dir.name)
        if named is None:
            raise ValueError(f"{self.dir.name}: not an old <model>_<map>_<timestamp> run folder")
        return named

    @cached_property
    def summary(self) -> dict[str, Any]:
        return json.loads((self.dir / "results" / "summary.json").read_text("utf-8"))

    @cached_property
    def locations(self) -> list[Round]:
        metadata = json.loads((self.dataset / self.folder["map"] / "metadata.json").read_text("utf-8"))
        bounds = Bounds.model_validate(metadata["bounds"])
        return [Round.parse(raw, bounds) for raw in metadata["images"]]

    def locate(self, cells: dict[str, str]) -> str:
        lat, lng = float(cells["lat_true"]), float(cells["lng_true"])
        found = [r.id for r in self.locations if abs(r.location.lat - lat) < SAME_POINT and abs(r.location.lng - lng) < SAME_POINT]
        if len(found) != 1:
            raise ValueError(f"{self.dir.name}: ({lat}, {lng}) is {len(found)} locations of {self.folder['map']}")
        return found[0]

    @cached_property
    def rows(self) -> list[Row]:
        with open(self.dir / "results" / "detailed.csv", newline="", encoding="utf-8") as f:
            return [Row(cells["location_id"] if "location_id" in cells else self.locate(cells), cells) for cells in csv.DictReader(f)]

    @property
    def model(self) -> str:
        return self.summary["model_id"]

    @property
    def median(self) -> Median:
        recorded = {
            key: [v for row in self.rows if (v := row.number(column)) is not None]
            for key, column in (("median_score", "score"), ("median_distance_km", "distance_km"))
        }
        fits = [
            rule
            for rule in (Median.HIGH, Median.MIDDLE)
            if all(self.summary.get(key) is None or isclose(rule(values), self.summary[key], rel_tol=SAME_SUM) for key, values in recorded.items())
        ]
        if not fits:
            raise ValueError(f"{self.dir.name}: the summary medians fit no median rule")
        return fits[0]

    @property
    def config(self) -> dict[str, Any]:
        return {
            "env": {"taskset": {"id": "geobench", "map": self.folder["map"]}},
            "model": self.model,
            "sampling": self.summary.get("sampling"),
            "report": {"median": self.median},
        }

    def reply(self, row: Row) -> str | None:
        output = self.dir / "output" / f"{row.id}.txt"
        return output.read_text("utf-8") if output.is_file() else None

    def usage(self, row: Row) -> dict[str, int] | None:
        raw = self.dir / "json" / f"{row.id}.json"
        return provider_usage(json.loads(raw.read_bytes().decode("utf-8", "replace"))) if raw.is_file() else None

    @property
    def episodes(self) -> list[dict[str, Any]]:
        started = datetime.strptime(self.folder["started"], STARTED).replace(tzinfo=UTC).timestamp()
        episodes = []
        for row in self.rows:
            task = {"type": "GeoTask", "data": {"id": f"{self.folder['map']}/{row.id}", "round": row.round}}
            reply = self.reply(row)
            trace = {
                "task": task,
                "agent": {"config": {"model": self.model}},
                "nodes": [] if reply is None else [{"message": {"role": "assistant", "content": reply}, "sampled": True}],
                "info": {"items": {row.id: row.item.model_dump()}},
                "timing": {"start": started},
                **({"usage": spent} if (spent := self.usage(row)) else {}),
            }
            episodes.append({"task": task, "traces": [trace]})
        return episodes

    def write(self) -> None:
        (self.dir / "configs" / "resolved").mkdir(parents=True, exist_ok=True)
        (self.dir / "configs" / "resolved" / "run.json").write_text(json.dumps(self.config, indent=2) + "\n", "utf-8")
        lines = (json.dumps(episode, ensure_ascii=False) + "\n" for episode in self.episodes)
        (self.dir / "traces.jsonl").write_text("".join(lines), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m geobench.legacy",
        description="Translate old-harness run folders into traces.jsonl and configs/resolved/run.json beside their old files.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="old run folders, each with results/summary.json")
    parser.add_argument("--dataset", type=Path, default=GeoTasksetConfig().dataset, help="maps folder, for old CSVs without location ids")
    args = parser.parse_args()
    for dir in args.runs:
        LegacyRun(dir, args.dataset).write()
    print(f"translated {len(args.runs)} runs")


if __name__ == "__main__":
    main()
