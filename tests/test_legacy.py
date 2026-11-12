import csv
import json
from math import isclose
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import verifiers.v1 as vf
from conftest import CATALOG, IMAGES, MAP, ROOT, write_map
from verifiers.v1.utils.trace_store import read_episodes

from geobench.leaderboard import Entry, Status, leaderboard
from geobench.legacy import LegacyRun

RESPONSES = ROOT / "responses"
SUMMARY_TO_ENTRY = {"test": "map", "total_samples": "n", "model_id": "model"}
NOT_SHOWN = {"model", "sampling"}
COLUMNS = ["location_id", "country_true", "lat_true", "lng_true", "refused", "error_message", "country_guess", "lat_guess", "lng_guess", "distance_km", "score", "country_correct"]
ANSWERED = [
    ["1", "United States of America (the)", 39.4817, -98.2263, False, "", "United States", 40.0, -90.0, 308.6, 4233.0, True],
    ["2", "Kenya", 0.6142, 37.3519, False, "", "Tanzania", -3.0, 36.0, 285.1, 4300.0, False],
    ["3", "Japan", 35.6812362, 139.7671248, False, "", "Japan", 35.0, 139.0, 102.4, 4700.0, False],
]
REFUSED = ["4", "Japan", 35.6812362, 139.7671248, True, "Format error: no answer", "", "", "", "", "", ""]
SUMMARY = {
    "model": "Lab Model 1",
    "test": MAP,
    "n": 4,
    "country_success_rate": 0.25,
    "refusal_rate": 0.25,
    "average_distance_km": (308.6 + 285.1 + 102.4) / 3,
    "average_score": (4233.0 + 4300.0 + 4700.0) / 3,
    "median_distance_km": 285.1,
    "median_score": 4300.0,
    "model_id": "lab/model-1",
}


def write_old_run(dir: Path, rows: list[list[Any]], summary: dict[str, Any], columns: list[str] = COLUMNS) -> Path:
    (dir / "results").mkdir(parents=True)
    (dir / "output").mkdir()
    with open(dir / "results" / "detailed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows([cell for column, cell in zip(COLUMNS, row) if column in columns] for row in rows)
    (dir / "results" / "summary.json").write_text(json.dumps(summary))
    for row in rows[:3]:
        (dir / "output" / f"{row[0]}.txt").write_text(f"Reasoning.\r\ncountry: {row[6]}\r\nlat: {row[7]}\r\nlng: {row[8]}")
    return dir


def disagreements(summary: dict[str, Any], shown: dict[str, Any]) -> set[str]:
    def same(key: str, recorded: Any) -> bool:
        value = shown[SUMMARY_TO_ENTRY.get(key, key)]
        if key == "test":
            return PurePosixPath(recorded).name == value
        if isinstance(recorded, float) and value is not None:
            return isclose(recorded, value, rel_tol=1e-12)
        return recorded == value

    return {key for key, recorded in summary.items() if key not in NOT_SHOWN and not same(key, recorded)}


def test_an_old_run_translates_to_its_summary(tmp_path: Path) -> None:
    dir = write_old_run(tmp_path / f"Lab Model 1_{MAP}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], SUMMARY)
    old = LegacyRun(dir, tmp_path / "dataset")
    old.write()

    assert sorted(path.name for path in dir.iterdir()) == ["configs", "output", "results", "traces.jsonl"]
    shown = Entry.read(tmp_path, dir).as_json(CATALOG)
    assert disagreements(SUMMARY, shown) == set()
    assert (shown["status"], shown["date"], shown["name"], shown["lab_name"], shown["released"]) == (Status.PUBLIC, "2025-05-22", "Lab Model 1", "Lab", "2025-01-14")
    assert (shown["deprecated"], shown["thinking"], shown["effort"], shown["search"]) == (False, False, None, False)
    assert old.config["sampling"] is None
    assert old.config["report"] == {"median": "high"}

    [episode, *_, refused] = read_episodes(dir, vf.Trace)
    [trace] = episode.traces
    assert trace.last_reply == "Reasoning.\ncountry: United States\nlat: 40.0\nlng: -90.0"
    assert trace.info["items"]["1"] == {
        "score": 4233.0 / 5000,
        "passed": True,
        "kind": None,
        "guess": {"country": "United States", "coordinates": {"lat": 40.0, "lng": -90.0}},
        "refused": False,
        "points": 4233.0,
        "distance_km": 308.6,
    }
    assert refused.traces[0].nodes == []
    assert refused.traces[0].info["items"]["4"] == {"score": 0.0, "passed": False, "kind": None, "guess": None, "refused": True, "error_message": "Format error: no answer"}


@pytest.mark.parametrize(("median_score", "median"), [(4700.0, "high"), (4500.0, "middle")])
def test_the_median_rule_is_the_one_the_summary_used(tmp_path: Path, median_score: float, median: str) -> None:
    summary = {"median_score": median_score, "median_distance_km": None, "model_id": "lab/model-1"}
    dir = write_old_run(tmp_path / f"Lab Model 1_{MAP}_2025-03-25T14_35_56", ANSWERED[1:], summary)
    assert LegacyRun(dir, tmp_path / "dataset").config["report"] == {"median": median}


def test_an_old_csv_without_location_ids_finds_them_in_the_dataset(tmp_path: Path) -> None:
    dataset = write_map(tmp_path / "dataset")
    rows = [ANSWERED[2], ANSWERED[0]]
    dir = write_old_run(tmp_path / f"LabModel1_{MAP}_2025-02-27T19_58_15", rows, SUMMARY | {"n": 2}, COLUMNS[1:])
    episodes = LegacyRun(dir, dataset).episodes
    assert [episode["task"]["data"]["id"] for episode in episodes] == [f"{MAP}/3", f"{MAP}/1"]
    assert episodes[0]["traces"][0]["nodes"][0]["message"]["content"].endswith("lng: 139.0")
    assert episodes[0]["task"]["data"]["round"] == {"id": "3", "country": IMAGES[2]["country"], "location": {"lat": IMAGES[2]["lat"], "lng": IMAGES[2]["lng"]}}


def test_the_leaderboard_shelves_runs_by_folder_and_sorts_them(tmp_path: Path) -> None:
    shelved = [
        ("", "lab/model-2", "acw-02-25-25", {}),
        ("", "lab/model-1", "acw-02-25-25", {}),
        (".hidden", "lab/model-1", "avw-03-27-25", {}),
        (".incomplete", "model-1", "acw-02-25-25", {}),
    ]
    for n, (shelf, model, map, changes) in enumerate(shelved):
        summary = {key: value for key, value in (SUMMARY | {"model_id": model, "test": map} | changes).items() if value is not None}
        LegacyRun(write_old_run(tmp_path / shelf / f"Lab Model {n}_{map}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], summary), tmp_path).write()
    board = leaderboard(tmp_path, CATALOG)
    assert [(r["map"], r["model"], r["status"], r["deprecated"]) for r in board["runs"]] == [
        ("acw-02-25-25", "lab/model-1", Status.INCOMPLETE, False),
        ("acw-02-25-25", "lab/model-1", Status.PUBLIC, False),
        ("acw-02-25-25", "lab/model-2", Status.PUBLIC, True),
        ("avw-03-27-25", "lab/model-1", Status.HIDDEN, False),
    ]
    assert json.dumps(board) == json.dumps(leaderboard(tmp_path, CATALOG))


@pytest.mark.parametrize(
    ("sampling", "thinking", "effort", "search"),
    [
        ({"thinking": {"type": "enabled"}}, True, None, False),
        ({"reasoning_effort": "high"}, True, "high", False),
        ({"tools": [{"google_search": {}}]}, False, None, True),
    ],
)
def test_a_runs_sampling_says_whether_it_thought_and_searched(tmp_path: Path, sampling: dict[str, Any], thinking: bool, effort: str | None, search: bool) -> None:
    dir = write_old_run(tmp_path / f"Lab Model 1_{MAP}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], SUMMARY | {"sampling": sampling})
    LegacyRun(dir, tmp_path).write()
    shown = Entry.read(tmp_path, dir).as_json(CATALOG)
    assert (shown["thinking"], shown["effort"], shown["search"]) == (thinking, effort, search)


@pytest.mark.parametrize(
    ("reply", "spent"),
    [
        ({"usage": {"input_tokens": 10, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 1, "output_tokens": 7}}, (16, 7)),
        ({"usage": {"input_tokens": 10, "output_tokens": 7, "total_tokens": 17}}, (10, 7)),
        ({"usage": {"prompt_tokens": 10, "completion_tokens": 7}}, (10, 7)),
        ({"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 4}}, (10, 7)),
    ],
)
def test_saved_provider_replies_give_each_trace_its_tokens_and_the_run_its_cost(tmp_path: Path, reply: dict[str, Any], spent: tuple[int, int]) -> None:
    dir = write_old_run(tmp_path / f"Lab Model 1_{MAP}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], SUMMARY)
    (dir / "json").mkdir()
    for row in ANSWERED:
        (dir / "json" / f"{row[0]}.json").write_text(json.dumps(reply))
    LegacyRun(dir, tmp_path).write()
    shown = Entry.read(tmp_path, dir).as_json(CATALOG)
    tokens = (3 * spent[0], 3 * spent[1])
    assert (shown["input_tokens"], shown["output_tokens"]) == tokens
    assert isclose(shown["cost"], (tokens[0] * 2 + tokens[1] * 8) / 1_000_000)

