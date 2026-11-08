import json
import re
from importlib.resources import files

ALIASES: dict[str, str] = {
    name.casefold(): canonical
    for canonical, aliases in json.loads(files(__package__).joinpath("countries.json").read_text("utf-8")).items()
    for name in (canonical, *aliases)
}
PARTS = re.compile(r"[,()]")


def canonical_country(name: str) -> str | None:
    spaced = " ".join(name.split())
    key = spaced.casefold()
    if not key:
        return None
    if key in ALIASES:
        return ALIASES[key]
    named = {ALIASES[part] for part in map(str.strip, PARTS.split(key)) if part in ALIASES}
    return named.pop() if len(named) == 1 else spaced


def same_country(a: str, b: str) -> bool:
    first, second = canonical_country(a), canonical_country(b)
    return first is not None and second is not None and first.casefold() == second.casefold()
