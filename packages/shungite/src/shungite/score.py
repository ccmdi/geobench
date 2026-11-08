from collections.abc import Callable
from dataclasses import dataclass

Score = Callable[[float], float]


@dataclass(frozen=True)
class Bands:
    steps: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        limits = [limit for limit, _ in self.steps]
        if not limits or any(a >= b for a, b in zip(limits, limits[1:])):
            raise ValueError(f"band limits must be non-empty and strictly ascending, got {limits}")

    def __call__(self, km: float) -> float:
        return next((score for limit, score in self.steps if km <= limit), 0.0)


def bands(*steps: tuple[float, float]) -> Bands:
    return Bands(steps)


def geoguessr(scale: float) -> Score:
    def score(km: float) -> float:
        meters = km / 1000 * 1_000_000
        return 5000.0 if meters <= 25 else float(round(5000 * 0.99866017 ** (meters / scale)))

    return score
