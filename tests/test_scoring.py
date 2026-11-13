import pytest
import verifiers.v1 as vf
from conftest import MAP, final, replied
from haversine import Direction, inverse_haversine

from geobench.taskset import GeoTask
from shungite import Coordinates

def offset(point: Coordinates, km: float) -> Coordinates:
    lat, lng = inverse_haversine((point.lat, point.lng), km, Direction.NORTH)
    return Coordinates(lat=lat, lng=lng)


async def scored(task: GeoTask, reply: str) -> vf.Trace:
    trace = replied(task, reply)
    await task.score(trace)
    return trace


async def test_the_reference_scores_full_points(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/1"]
    location = task.data.round.location
    trace = await scored(task, final("United States", location))
    assert trace.reward == 1.0
    assert trace.metrics == {"score": 5000.0, "country_correct": 1.0, "answered": 1.0, "distance_km": 0.0}
    assert trace.info["items"] == {
        "1": {
            "score": 1.0,
            "passed": True,
            "kind": None,
            "guess": {"country": "United States", "coordinates": location.model_dump()},
            "refused": False,
            "points": 5000.0,
            "distance_km": 0.0,
        }
    }


async def test_a_far_guess_scores_on_the_map_curve(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/2"]
    trace = await scored(task, final("Kenya", offset(task.data.round.location, 1204.3488020420805)))
    assert trace.metrics["score"] == 2244.0
    assert trace.reward == pytest.approx(2244 / 5000)
    assert trace.metrics["distance_km"] == pytest.approx(1204.3488020420805, abs=1e-6)


async def test_an_unanswered_reply_scores_zero(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/1"]
    trace = await scored(task, "I cannot tell where this is.")
    assert trace.reward == 0.0
    assert trace.metrics == {"score": 0.0, "country_correct": 0.0, "answered": 0.0}
    assert trace.info["items"]["1"] == {"score": 0.0, "passed": False, "kind": None, "guess": None, "refused": True}


async def test_an_empty_reply_scores_zero(tasks: dict[str, GeoTask]) -> None:
    trace = await scored(tasks[f"{MAP}/1"], "")
    assert trace.reward == 0.0
    assert trace.metrics["answered"] == 0.0


async def test_a_round_passes_on_the_right_country_whatever_the_distance(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/2"]
    near = await scored(task, final("Tanzania", task.data.round.location))
    far = await scored(task, final("Kenya", offset(task.data.round.location, 3000.0)))
    assert (near.info["items"]["2"]["score"], near.info["items"]["2"]["passed"]) == (1.0, False)
    assert far.info["items"]["2"]["passed"] is True
    assert far.info["items"]["2"]["score"] < 0.5


@pytest.mark.parametrize(
    ("country", "correct"),
    [("United States", 1.0), ("USA", 1.0), ("united states of america", 1.0), ("Canada", 0.0)],
)
async def test_country_is_matched_through_aliases(tasks: dict[str, GeoTask], country: str, correct: float) -> None:
    task = tasks[f"{MAP}/1"]
    trace = await scored(task, final(country, task.data.round.location))
    assert trace.metrics["country_correct"] == correct


async def test_coordinates_without_a_country_score_but_miss_the_country(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/3"]
    location = task.data.round.location
    trace = await scored(task, f"lat: {location.lat}\nlng: {location.lng}")
    assert trace.reward == 1.0
    assert trace.metrics["country_correct"] == 0.0


async def test_the_last_block_is_graded(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/3"]
    location = task.data.round.location
    reply = final("Kenya", Coordinates(lat=0.0, lng=37.0)) + "\n\nOn reflection:\n" + final("Japan", location)
    trace = await scored(task, reply)
    assert trace.reward == 1.0
    assert trace.metrics["country_correct"] == 1.0


async def test_rescore_reproduces_the_score(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/2"]
    trace = replied(task, final("Tanzania", offset(task.data.round.location, 300.0)))
    trace.ok = True
    await task.score(trace)
    first = trace.model_dump(include={"rewards", "metrics", "info"})
    await task.rescore(trace)
    assert trace.model_dump(include={"rewards", "metrics", "info"}) == first
    assert 0.0 < trace.reward < 1.0
    assert trace.metrics["country_correct"] == 0.0


def answered(country: str, lat: float, lng: float) -> dict[str, object]:
    return {"country": country, "coordinates": {"lat": lat, "lng": lng}}


@pytest.mark.parametrize(
    ("reply", "guess", "points"),
    [
        pytest.param(
            "<thinking>\ncountry: South Korea\nlat: 37.5\nlng: 127.0\n</thinking>\n\ncountry: Japan\nlat: 35.0\nlng: 139.0",
            answered("Japan", 35.0, 139.0),
            4567.0,
            id="draft-in-thinking",
        ),
        pytest.param(
            "country: Japan or Korea\nlat: 36°N\nlng: somewhere east\n\ncountry: Japan\nlat: 35.5\nlng: 138.5",
            answered("Japan", 35.5, 138.5),
            4769.0,
            id="malformed-block-before-answer",
        ),
        pytest.param(
            "*   **Country**: Japan\n*   **Latitude**: 36.5° N\n*   **Longitude**: 139.5° E",
            answered("Japan", 36.5, 139.5),
            4669.0,
            id="bulleted-hemispheres",
        ),
        pytest.param(
            "country: Japan\nlat: 36.2\nlng: 138.3 (approximate for the Nagano region)",
            answered("Japan", 36.2, 138.3),
            4952.0,
            id="trailing-note",
        ),
        pytest.param("Approximate latitude: 36.0\nApproximate longitude: 138.0", None, 0.0, id="prose-labels"),
        pytest.param("I'm sorry, but I can't determine where this photo was taken.", None, 0.0, id="refusal"),
    ],
)
async def test_only_the_final_answer_block_scores(tasks: dict[str, GeoTask], reply: str, guess: dict[str, object] | None, points: float) -> None:
    trace = await scored(tasks[f"{MAP}/3"], reply)
    assert (trace.info["items"]["3"]["guess"], trace.metrics["score"]) == (guess, points)
