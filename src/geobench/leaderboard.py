import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Self
from urllib.parse import urlparse

from geobench.catalog import Catalog
from shungite.report import Prices, Result, Run, Setup

STARTED = "%Y-%m-%dT%H_%M_%S"


class Status(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"
    INCOMPLETE = "incomplete"


SHELVES = {".": Status.PUBLIC, ".hidden": Status.HIDDEN, ".incomplete": Status.INCOMPLETE}


def run_dirs(root: Path) -> Iterator[Path]:
    for shelf in SHELVES:
        if (root / shelf).is_dir():
            yield from sorted(dir for dir in (root / shelf).iterdir() if dir.is_dir() and not dir.name.startswith("."))


@dataclass(frozen=True)
class Entry:
    root: Path
    path: PurePosixPath
    config: dict[str, Any]

    @classmethod
    def read(cls, root: Path, dir: Path) -> Self:
        config = json.loads((dir / "configs" / "resolved" / "run.json").read_text("utf-8"))
        return cls(root, PurePosixPath(dir.relative_to(root).as_posix()), config)

    @cached_property
    def run(self) -> Run:
        return Run.read(self.root / self.path)

    @property
    def status(self) -> Status:
        return SHELVES[str(self.path.parent)]

    @property
    def map(self) -> str | None:
        return self.config["env"]["taskset"].get("map")

    @property
    def sampling(self) -> dict[str, Any]:
        return self.config.get("sampling") or {}

    @property
    def effort(self) -> str | None:
        return self.sampling.get("reasoning_effort")

    @property
    def thinking(self) -> bool:
        return self.effort is not None or (self.sampling.get("thinking") or {}).get("type") == "enabled"

    @property
    def search(self) -> bool:
        return any("google_search" in tool for tool in self.sampling.get("tools", []))

    @property
    def settings(self) -> dict[str, Any]:
        return Setup.of(self.config).settings or {}

    @property
    def started(self) -> str:
        return datetime.strptime(self.path.name.rpartition(f"_{self.map}_")[2], STARTED).replace(tzinfo=UTC).isoformat()

    @property
    def provider(self) -> str | None:
        return urlparse((self.config.get("client") or {}).get("base_url", "")).hostname

    def card(self, catalog: Catalog) -> dict[str, Any]:
        model = catalog[self.config["model"]]
        return {
            "run": str(self.path),
            "status": self.status,
            "map": self.map,
            "model": model.id,
            "name": model.name,
            "lab": model.lab,
            "lab_name": model.lab_name,
            "released": model.released,
            "deprecated": model.deprecated,
            "thinking": self.thinking,
            "effort": self.effort,
            "search": self.search,
            "temperature": self.sampling.get("temperature"),
            "max_tokens": self.sampling.get("max_tokens"),
        }

    def as_json(self, catalog: Catalog) -> dict[str, Any]:
        result = Result(self.run.setup, (self.run,))
        measures = result.measures
        tokens, cost = result.bill(Prices(model=catalog[result.setup.model].price))["model"]

        def measure(name: str, statistic: str) -> float | None:
            return getattr(measures[name], statistic) if name in measures else None

        return self.card(catalog) | {
            "date": next(iter(result.run_dates), None),
            "n": len(result.items),
            "country_success_rate": result.pass_rate,
            "refusal_rate": measure("refused", "mean"),
            "average_distance_km": measure("distance_km", "mean"),
            "average_score": measure("points", "mean"),
            "median_distance_km": measure("distance_km", "median"),
            "median_score": measure("points", "median"),
            "input_tokens": None if tokens is None else tokens.input,
            "output_tokens": None if tokens is None else tokens.output,
            "cost": cost,
        }


def leaderboard(root: Path, catalog: Catalog) -> dict[str, Any]:
    entries = [Entry.read(root, dir).as_json(catalog) for dir in run_dirs(root)]
    return {"runs": sorted(entries, key=lambda e: (e["map"] or "", e["model"], e["run"]))}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m geobench.leaderboard",
        description="Write the leaderboard JSON from every run under a responses folder and its .hidden/ and .incomplete/ shelves.",
    )
    parser.add_argument("root", type=Path, help="responses folder")
    parser.add_argument("output", type=Path, help="JSON file to write")
    args = parser.parse_args()
    args.output.write_text(json.dumps(leaderboard(args.root, Catalog.fetch()), indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
