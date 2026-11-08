from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from shungite import Bounds, Coordinates, Guess, geoguessr, read_guess, same_country

MAX_POINTS = 5000.0


class Round(BaseModel):
    model_config = ConfigDict(frozen=True)

    country: str
    location: Coordinates
    bounds: Bounds

    @classmethod
    def parse(cls, raw: dict[str, Any], bounds: Bounds) -> Self:
        return cls(country=raw["country"], location=Coordinates(lat=raw["lat"], lng=raw["lng"]), bounds=bounds)

    def grade(self, reply: str) -> "Grade":
        return Grade(self, read_guess(reply))


@dataclass(frozen=True)
class Grade:
    round: Round
    guess: Guess | None

    @property
    def km(self) -> float | None:
        return None if self.guess is None else self.round.location.km_to(self.guess.coordinates)

    @property
    def points(self) -> float:
        km = self.km
        return 0.0 if km is None else geoguessr(self.round.bounds.scale)(km)

    @property
    def country_correct(self) -> bool:
        return self.guess is not None and self.guess.country is not None and same_country(self.guess.country, self.round.country)

    @property
    def metrics(self) -> dict[str, float]:
        km = self.km
        distance = {} if km is None else {"distance_km": km}
        return {"score": self.points, "country_correct": float(self.country_correct), "answered": float(self.guess is not None)} | distance

    @property
    def verdict(self) -> dict[str, Any]:
        return {"guess": None if self.guess is None else self.guess.model_dump()} | self.metrics
