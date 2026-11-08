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
    assert trace.info["verdict"] == {
        "guess": {"country": "United States", "coordinates": location.model_dump()},
        "score": 5000.0,
        "country_correct": 1.0,
        "answered": 1.0,
        "distance_km": 0.0,
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
    assert trace.info["verdict"]["guess"] is None


async def test_an_empty_reply_scores_zero(tasks: dict[str, GeoTask]) -> None:
    trace = await scored(tasks[f"{MAP}/1"], "")
    assert trace.reward == 0.0
    assert trace.metrics["answered"] == 0.0


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
