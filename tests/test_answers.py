from dataclasses import replace
from pathlib import Path

from conftest import GRADED, IMAGES, MAP, write_run

from geobench.answers import Answer, answers


def test_answers_flatten_each_listed_runs_items_with_their_truth(tmp_path: Path) -> None:
    for shelf, model in ((".", "lab/model-1"), (".hidden", "lab/model-2")):
        write_run(tmp_path / shelf / f"{model.replace('/', '_')}_{MAP}_2025-05-22T12_52_37", model=model)
    shared = {"run": f"lab_model-1_{MAP}_2025-05-22T12_52_37", "map": MAP, "model": "lab/model-1", "reply": None, "reasoning": None}
    listed = list(answers(tmp_path))
    assert [answer.location for answer in listed] == ["1", "2", "3", "4"]
    assert [listed[0], listed[-1]] == [
        Answer(
            **shared,
            task=f"{MAP}/1",
            location="1",
            country="United States of America (the)",
            place="United States",
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
            place="Japan",
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


def test_an_answer_keeps_the_reply_and_reasoning_behind_its_guess(tmp_path: Path) -> None:
    write_run(tmp_path / f"lab_model-1_{MAP}_2025-05-22T12_52_37", [replace(GRADED[0], reply="country: United States", reasoning="corn")])
    [answer] = answers(tmp_path)
    assert (answer.reply, answer.reasoning) == ("country: United States", "corn")
