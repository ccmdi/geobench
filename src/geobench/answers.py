from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from geobench.leaderboard import Entry, Status, run_dirs
from shungite import Guess, Item
from shungite.report import Trial

@dataclass(frozen=True)
class Answer:
    run: str
    map: str | None
    model: str
    provider: str | None
    task: str
    location: str
    country: str
    lat: float
    lng: float
    guess_country: str | None
    guess_lat: float | None
    guess_lng: float | None
    refused: bool
    passed: bool
    points: float | None
    distance_km: float | None
    error: str | None

    @classmethod
    def of(cls, entry: Entry, trial: Trial, item: Item) -> Self:
        round, extra = trial.data["round"], item.model_extra or {}
        guess = None if extra.get("guess") is None else Guess.model_validate(extra["guess"])
        return cls(
            run=str(entry.path),
            map=entry.map,
            model=entry.run.setup.model,
            provider=entry.board.get("provider"),
            task=trial.task,
            location=round["id"],
            country=round["country"],
            lat=round["location"]["lat"],
            lng=round["location"]["lng"],
            guess_country=None if guess is None else guess.country,
            guess_lat=None if guess is None else guess.coordinates.lat,
            guess_lng=None if guess is None else guess.coordinates.lng,
            refused=extra["refused"],
            passed=bool(item.passed),
            points=extra.get("points"),
            distance_km=extra.get("distance_km"),
            error=extra.get("error_message"),
        )


def answers(root: Path) -> Iterator[Answer]:
    for dir in run_dirs(root):
        entry = Entry.read(root, dir)
        if entry.status is Status.PUBLIC:
            yield from (Answer.of(entry, trial, item) for trial in entry.run.trials for item in (trial.items or {}).values())
