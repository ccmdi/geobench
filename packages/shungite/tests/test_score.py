import pytest

from shungite import Bounds, bands, geoguessr

STEPS = bands((0.1, 1.0), (0.5, 0.8), (1.0, 0.5), (10.0, 0.2))

SCALES = {
    "acw": 2485.084631421639,
    "avw": 2485.810918492928,
    "ext-urban": 1873.6252126849456,
    "geoguessr-2069": 1857.783934490492,
    "rural-world": 1992.0045023950618,
}


@pytest.mark.parametrize(
    ("km", "expected"),
    [
        (0.0, 1.0),
        (0.1, 1.0),
        (0.1000001, 0.8),
        (0.5, 0.8),
        (1.0, 0.5),
        (1.0000001, 0.2),
        (10.0, 0.2),
        (10.0000001, 0.0),
        (20000.0, 0.0),
    ],
)
def test_bands_limits_are_inclusive(km: float, expected: float) -> None:
    assert STEPS(km) == expected


def test_bands_expose_steps() -> None:
    assert STEPS.steps == ((0.1, 1.0), (0.5, 0.8), (1.0, 0.5), (10.0, 0.2))


@pytest.mark.parametrize("steps", [(), ((1.0, 1.0), (0.5, 0.5)), ((1.0, 1.0), (1.0, 0.5))])
def test_bands_reject_unordered_steps(steps: tuple[tuple[float, float], ...]) -> None:
    with pytest.raises(ValueError):
        bands(*steps)


def test_bounds_scale() -> None:
    bounds = Bounds.model_validate({"min": {"lat": -55.0, "lng": -170.0}, "max": {"lat": 80.0, "lng": 178.0}})
    assert bounds.scale == pytest.approx(2015.3, abs=0.05)


@pytest.mark.parametrize(
    ("name", "km", "expected"),
    [
        ("acw", 1204.3488020420805, 2611),
        ("acw", 820.3581972416703, 3212),
        ("acw", 692.1338884349159, 3442),
        ("rural-world", 922.7271975694032, 2687),
        ("rural-world", 90.21220927427461, 4705),
        ("ext-urban", 1.8132782935287581, 4994),
        ("ext-urban", 221.39521552250372, 4267),
        ("avw", 22.0058341522011, 4941),
        ("avw", 1049.566129881136, 2839),
        ("geoguessr-2069", 4.086764860149058, 4985),
        ("geoguessr-2069", 1374.6369352585073, 1854),
    ],
)
def test_geoguessr_matches_recorded_scores(name: str, km: float, expected: int) -> None:
    assert geoguessr(SCALES[name])(km) == float(expected)


def test_geoguessr_25_metres_is_perfect() -> None:
    score = geoguessr(1.0)
    assert score(0.0) == 5000.0
    assert score(0.025) == 5000.0
    assert score(0.026) == float(round(5000 * 0.99866017**26))
    assert geoguessr(SCALES["acw"])(0.04336056466641199) == 5000.0
