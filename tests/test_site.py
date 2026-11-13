import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import CATALOG, MAP, write_run

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
    for shelf, name in ((".", "Lab Model 1"), (".hidden", "Lab Model 3")):
        write_run(tmp_path / shelf / f"{name}_{MAP}_2025-05-22T12_52_37")
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
