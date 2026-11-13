import pytest

from shungite import bands

STEPS = bands((0.1, 1.0), (0.5, 0.8), (1.0, 0.5), (10.0, 0.2))


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
