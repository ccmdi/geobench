import json
from collections.abc import Iterator
from math import isclose
from pathlib import Path
from typing import Any

import pytest
from conftest import CATALOG, MAP
from test_legacy import ANSWERED, REFUSED, RESPONSES, SUMMARY, write_old_run

from geobench.catalog import Catalog
from geobench.leaderboard import Status, leaderboard
from geobench.legacy import LegacyRun
from geobench.site import site

RUN = f"Lab Model 1_{MAP}_2025-05-22T12_52_37"
COORDINATES = ("lat", "lng", "location", "coordinates", "guess")


def keys(value: Any) -> Iterator[str]:
    match value:
        case dict():
            for key, item in value.items():
                yield key
                yield from keys(item)
        case list():
            for item in value:
                yield from keys(item)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for shelf, model in ((".", "Lab Model 1"), (".hidden", "Lab Model 3")):
        LegacyRun(write_old_run(tmp_path / shelf / f"{model}_{MAP}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], SUMMARY | {"model": model}), tmp_path).write()
    return tmp_path


def test_the_site_shows_public_runs_with_their_distances_and_countries(root: Path) -> None:
    assert site(root, CATALOG) == {
        "maps": [{"id": MAP, "n": 4}],
        "runs": [
            {
                "run": RUN,
                "map": MAP,
                "model": "lab/model-1",
                "name": "Lab Model 1",
                "lab": "lab",
                "lab_name": "Lab",
                "released": "2025-01-14",
                "deprecated": False,
                "thinking": False,
                "effort": None,
                "search": False,
                "temperature": None,
                "max_tokens": None,
                "date": "2025-05-22",
                "n": 4,
                "country_success_rate": 0.25,
                "refusal_rate": 0.25,
                "average_distance_km": 232.0,
                "average_score": 4411.0,
                "median_distance_km": 285.1,
                "median_score": 4300.0,
                "input_tokens": None,
                "output_tokens": None,
                "cost": None,
                "distances_km": [102.4, 285.1, 308.6],
                "countries": [
                    {"country": "Japan", "n": 2, "country_success_rate": 0.0, "average_score": 4700.0},
                    {"country": "Kenya", "n": 1, "country_success_rate": 0.0, "average_score": 4300.0},
                    {"country": "United States", "n": 1, "country_success_rate": 1.0, "average_score": 4233.0},
                ],
            }
        ],
    }


def test_no_key_in_the_site_carries_coordinates_or_location_ids(root: Path) -> None:
    assert not [key for key in keys(site(root, CATALOG)) if any(part in key for part in COORDINATES)]


def test_the_site_builds_the_same_twice(root: Path) -> None:
    assert json.dumps(site(root, CATALOG)) == json.dumps(site(root, CATALOG))


@pytest.mark.skipif(not RESPONSES.is_dir(), reason="responses/ is private and not in this clone")
def test_every_public_run_on_the_site_matches_the_leaderboard() -> None:
    catalog = Catalog.fetch()
    board = {run["run"]: run for run in leaderboard(RESPONSES, catalog)["runs"] if run["status"] is Status.PUBLIC}
    shown = site(RESPONSES, catalog)
    assert [run["run"] for run in shown["runs"]] == list(board)
    assert not [key for key in keys(shown) if any(part in key for part in COORDINATES)]
    for run in shown["runs"]:
        expected = board[run["run"]]
        countries = run["countries"]
        assert run["n"] == expected["n"] == sum(country["n"] for country in countries)
        for rate in ("country_success_rate", "refusal_rate"):
            assert isclose(run[rate], expected[rate], abs_tol=5e-4)
        passed = sum(country["country_success_rate"] * country["n"] for country in countries)
        assert isclose(passed, expected["country_success_rate"] * expected["n"], abs_tol=5e-4 * run["n"])
        assert len(run["distances_km"]) == round(run["n"] * (1 - expected["refusal_rate"]))
        assert all(country["country"] != country["country"].casefold() for country in countries)
        assert [country["country"] for country in countries] == sorted({country["country"] for country in countries})
