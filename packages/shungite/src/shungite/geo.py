from typing import Self

from haversine import haversine
from pydantic import BaseModel, ConfigDict, Field

DIAGONAL_TO_SCALE = 7.458421


class Coordinates(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)

    def km_to(self, other: Self) -> float:
        return haversine((self.lat, self.lng), (other.lat, other.lng))


class Bounds(BaseModel):
    model_config = ConfigDict(frozen=True)

    min: Coordinates
    max: Coordinates

    @property
    def scale(self) -> float:
        return self.min.km_to(self.max) / DIAGONAL_TO_SCALE
