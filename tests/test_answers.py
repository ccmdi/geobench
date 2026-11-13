from pathlib import Path

from conftest import IMAGES, MAP
from test_legacy import ANSWERED, REFUSED, SUMMARY, write_old_run

from geobench.answers import Answer, answers
from geobench.legacy import LegacyRun


def test_answers_flatten_each_listed_runs_items_with_their_truth(tmp_path: Path) -> None:
    for shelf, model in ((".", "Lab Model 1"), (".hidden", "Lab Model 2")):
        dir = write_old_run(tmp_path / shelf / f"{model}_{MAP}_2025-05-22T12_52_37", [*ANSWERED, REFUSED], SUMMARY | {"model": model})
        LegacyRun(dir, tmp_path).write()
    run = f"Lab Model 1_{MAP}_2025-05-22T12_52_37"
    shared = {"run": run, "map": MAP, "model": "lab/model-1"}
    listed = list(answers(tmp_path))
    assert [answer.location for answer in listed] == ["1", "2", "3", "4"]
    assert [listed[0], listed[-1]] == [
        Answer(
            **shared,
            task=f"{MAP}/1",
            location="1",
            country="United States of America (the)",
            lat=IMAGES[0]["lat"],
            lng=IMAGES[0]["lng"],
            guess_country="United States",
            guess_lat=40.0,
            guess_lng=-90.0,
            refused=False,
            passed=True,
            points=4233.0,
            distance_km=308.6,
            error=None,
        ),
        Answer(
            **shared,
            task=f"{MAP}/4",
            location="4",
            country="Japan",
            lat=IMAGES[2]["lat"],
            lng=IMAGES[2]["lng"],
            guess_country=None,
            guess_lat=None,
            guess_lng=None,
            refused=True,
            passed=False,
            points=None,
            distance_km=None,
            error="Format error: no answer",
        ),
    ]
