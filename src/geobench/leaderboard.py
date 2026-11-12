import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Self

from geobench.catalog import Catalog
from shungite.report import Prices, Result, Run


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
    path: PurePosixPath
    config: dict[str, Any]
    run: Run

    @classmethod
    def read(cls, root: Path, dir: Path) -> Self:
        config = json.loads((dir / "configs" / "resolved" / "run.json").read_text("utf-8"))
        return cls(PurePosixPath(dir.relative_to(root).as_posix()), config, Run.read(dir))

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

    def as_json(self, catalog: Catalog) -> dict[str, Any]:
        result = Result(self.run.setup, (self.run,))
        measures = result.measures
        model = catalog[result.setup.model]
        tokens, cost = result.bill(Prices(model=model.price))["model"]

        def measure(name: str, statistic: str) -> float | None:
            return getattr(measures[name], statistic) if name in measures else None

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
