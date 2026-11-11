import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

PARTS = re.compile(r"[,()]")
TABLE: dict[str, dict[str, Any]] = json.loads(files(__package__).joinpath("countries.json").read_text("utf-8"))


def spellings() -> Iterator[tuple[str, str, str]]:
    for country, entry in TABLE.items():
        yield from ((name, country, country) for name in (country, *entry["aliases"]))
        for place, aliases in entry.get("territories", {}).items():
            yield from ((name, country, place) for name in (place, *aliases))


@dataclass(frozen=True)
class Names:
    table: dict[str, str]

    def __call__(self, name: str) -> str | None:
        spaced = " ".join(name.split())
        key = spaced.casefold()
        if not key:
            return None
        if key in self.table:
            return self.table[key]
        named = {self.table[part] for part in map(str.strip, PARTS.split(key)) if part in self.table}
        return named.pop() if len(named) == 1 else spaced


canonical_country = Names({name.casefold(): country for name, country, _ in spellings()})
place_name = Names({name.casefold(): place for name, _, place in spellings()})


def same_country(a: str, b: str) -> bool:
    first, second = canonical_country(a), canonical_country(b)
    return first is not None and second is not None and first.casefold() == second.casefold()
