from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Unit = Annotated[float, Field(ge=0, le=1)]


class Item(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    score: Unit | None
    passed: bool | None
    kind: str | None = None

    @model_validator(mode="after")
    def undecided_as_a_whole(self) -> Self:
        if (self.score is None) != (self.passed is None):
            raise ValueError(f"an item is decided (score and passed) or undecided (neither), got {self!r}")
        return self

    @property
    def decided(self) -> bool:
        return self.passed is not None
