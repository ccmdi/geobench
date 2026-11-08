import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from shungite.geo import Coordinates

FIELD = re.compile(
    r"^[^a-z\n]*?(?P<key>country|latitude|lat|longitude|long|lon|lng)[\s*_`'\"]*:(?P<value>.*)$",
    re.IGNORECASE | re.MULTILINE,
)
NUMBER = re.compile(
    r"\s*(?:(?:approximately|approx\.?|about|around|roughly|circa|~|≈)\s*)?"
    r"(?P<sign>[-+]?)\s*(?P<magnitude>\d+(?:\.\d+)?|\.\d+)(?!\d|[.,]\d)"
    r"\s*(?:°|º|degrees?\b|deg\b)?+(?!\s*\d)\s*(?P<hemisphere>north|south|east|west|[nsew])?(?![a-z])",
    re.IGNORECASE,
)
MARKUP = re.compile(r"<[^>]*>|[*_`]")


class Guess(BaseModel):
    model_config = ConfigDict(frozen=True)

    country: str | None
    coordinates: Coordinates


@dataclass(frozen=True)
class Axis:
    limit: float
    positive: str
    negative: str

    def __call__(self, value: str) -> float | None:
        number = NUMBER.match(MARKUP.sub("", value))
        if number is None:
            return None
        magnitude, sign, hemisphere = float(number["magnitude"]), number["sign"], number["hemisphere"]
        if hemisphere is None:
            negative = sign == "-"
        elif (side := hemisphere[0].upper()) in (self.positive, self.negative):
            negative = side == self.negative
            if sign and negative != (sign == "-"):
                return None
        else:
            return None
        degrees = -magnitude if negative else magnitude
        return degrees if abs(degrees) <= self.limit else None


def country_text(value: str) -> str | None:
    return MARKUP.sub("", value).strip(" \t\r.,;\"'") or None


LAT = Axis(90, "N", "S")
LNG = Axis(180, "E", "W")
FIELDS: dict[str, tuple[str, Callable[[str], object]]] = {
    "country": ("country", country_text),
    "lat": ("lat", LAT),
    "latitude": ("lat", LAT),
    "lng": ("lng", LNG),
    "lon": ("lng", LNG),
    "long": ("lng", LNG),
    "longitude": ("lng", LNG),
}


def guesses(reply: str) -> Iterator[Guess]:
    block: dict[str, object] = {}
    for line in FIELD.finditer(reply):
        name, read = FIELDS[line["key"].lower()]
        value = read(line["value"])
        if value is None:
            continue
        if name in block or name == "country":
            block = {}
        block[name] = value
        if "lat" in block and "lng" in block:
            yield Guess.model_validate(
                {"country": block.get("country"), "coordinates": {"lat": block["lat"], "lng": block["lng"]}}
            )
            block = {}


def read_guess(reply: str) -> Guess | None:
    final = None
    for final in guesses(reply):
        pass
    return final
