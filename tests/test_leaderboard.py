import json
from math import isclose
from pathlib import Path
from typing import Any

import pytest
from conftest import CATALOG, GRADED, MAP, write_run

from geobench.leaderboard import Entry, Status, leaderboard
from shungite.report import Median

RUN = f"Lab Model 1_{MAP}_2025-05-22T12_52_37"


def shown(root: Path, dir: Path) -> dict[str, Any]:
    return Entry.read(root, dir).as_json(CATALOG)


def test_the_leaderboard_shelves_runs_by_folder_and_sorts_them(tmp_path: Path) -> None:
    shelved = [
        ("", "lab/model-2", "acw-02-25-25"),
        ("", "lab/model-1", "acw-02-25-25"),
        (".hidden", "lab/model-1", "avw-03-27-25"),
        (".incomplete", "model-1", "acw-02-25-25"),
    ]
    for n, (shelf, model, map) in enumerate(shelved):
        write_run(tmp_path / shelf / f"Lab Model {n}_{map}_2025-05-22T12_52_37", model=model, map=map)
    board = leaderboard(tmp_path, CATALOG)
    assert [(r["map"], r["model"], r["status"], r["deprecated"]) for r in board["runs"]] == [
        ("acw-02-25-25", "lab/model-1", Status.INCOMPLETE, False),
        ("acw-02-25-25", "lab/model-1", Status.PUBLIC, False),
        ("acw-02-25-25", "lab/model-2", Status.PUBLIC, True),
        ("avw-03-27-25", "lab/model-1", Status.HIDDEN, False),
    ]
    assert json.dumps(board) == json.dumps(leaderboard(tmp_path, CATALOG))


@pytest.mark.parametrize(("median", "median_score"), [(Median.HIGH, 4700.0), (Median.MIDDLE, 4500.0)])
def test_the_median_rule_is_the_one_the_run_records(tmp_path: Path, median: Median, median_score: float) -> None:
    dir = write_run(tmp_path / RUN, GRADED[1:3], median=median)
    assert shown(tmp_path, dir)["median_score"] == median_score


@pytest.mark.parametrize(
    ("sampling", "thinking", "effort", "search"),
    [
        ({"thinking": {"type": "enabled"}}, True, None, False),
        ({"reasoning_effort": "high"}, True, "high", False),
        ({"tools": [{"google_search": {}}]}, False, None, True),
    ],
)
def test_a_runs_sampling_says_whether_it_thought_and_searched(tmp_path: Path, sampling: dict[str, Any], thinking: bool, effort: str | None, search: bool) -> None:
    entry = shown(tmp_path, write_run(tmp_path / RUN, sampling=sampling))
    assert (entry["thinking"], entry["effort"], entry["search"]) == (thinking, effort, search)


def test_each_traces_tokens_add_up_to_the_runs_cost(tmp_path: Path) -> None:
    entry = shown(tmp_path, write_run(tmp_path / RUN, usage={"prompt_tokens": 10, "completion_tokens": 7}))
    assert (entry["input_tokens"], entry["output_tokens"]) == (40, 28)
    assert isclose(entry["cost"], (40 * 2 + 28 * 8) / 1_000_000)
