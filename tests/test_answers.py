from pathlib import Path

from conftest import IMAGES, MAP, write_run

from geobench.answers import Answer, answers


def test_answers_flatten_each_listed_runs_items_with_their_truth(tmp_path: Path) -> None:
    for shelf, model in ((".", "lab/model-1"), (".hidden", "lab/model-2")):
        write_run(tmp_path / shelf / f"{model.replace('/', '_')}_{MAP}_2025-05-22T12_52_37", model=model)
    shared = {"run": f"lab_model-1_{MAP}_2025-05-22T12_52_37", "map": MAP, "model": "lab/model-1"}
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
