import argparse
import json
from collections.abc import Iterable
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from statistics import fmean, mode
from typing import Any

from geobench.answers import Answer, answers
from geobench.leaderboard import Status, leaderboard
from shungite.country import canonical_country
from shungite.report import average

DIGITS = {"rate": 3, "score": 2, "km": 1}


def rounded(value: Any, digits: int | None = None) -> Any:
    match value:
        case dict():
            return {key: rounded(item, DIGITS.get(key.rpartition("_")[2])) for key, item in value.items()}
        case list():
            return [rounded(item, digits) for item in value]
        case float() if digits is not None:
            return round(value, digits)
        case _:
            return value


def country(name: str | None, rows: list[Answer]) -> dict[str, Any]:
    return {
        "country": name,
        "n": len(rows),
        "country_success_rate": fmean(row.passed for row in rows),
        "average_score": average(row.points for row in rows if row.points is not None),
    }


def breakdown(rows: Iterable[Answer]) -> dict[str, Any]:
    named = sorted(((canonical_country(row.country), row) for row in rows), key=itemgetter(0))
    return {
        "distances_km": sorted(row.distance_km for _, row in named if row.distance_km is not None),
        "countries": [country(name, [row for _, row in group]) for name, group in groupby(named, key=itemgetter(0))],
    }


def site(root: Path) -> dict[str, Any]:
    public = [{key: value for key, value in run.items() if key != "status"} for run in leaderboard(root)["runs"] if run["status"] is Status.PUBLIC]
    rows = {run: list(group) for run, group in groupby(answers(root), key=lambda answer: answer.run)}
    return rounded({
        "maps": [{"id": map, "n": mode(run["n"] for run in group)} for map, group in groupby(public, key=itemgetter("map"))],
        "runs": [run | breakdown(rows[run["run"]]) for run in public],
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m geobench.site",
        description="Write the website's JSON of public run aggregates, with no coordinates or location ids.",
    )
    parser.add_argument("root", type=Path, help="responses folder")
    parser.add_argument("output", type=Path, help="JSON file to write")
    args = parser.parse_args()
    args.output.write_text(json.dumps(site(args.root), ensure_ascii=False, separators=(",", ":")) + "\n", "utf-8")


if __name__ == "__main__":
    main()
