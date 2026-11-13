import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Self

import httpx

from shungite.report import Price

MODELS_URL = "https://models.dev/models.json"
PROVIDERS_URL = "https://models.dev/api.json"
DEPRECATED = "deprecated"


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    lab: str
    lab_name: str
    released: str
    deprecated: bool
    price: Price | None


@dataclass(frozen=True)
class Catalog:
    models: dict[str, dict[str, Any]]
    providers: dict[str, dict[str, Any]]

    @classmethod
    def fetch(cls) -> Self:
        with httpx.Client(timeout=30, follow_redirects=True) as http:
            models, providers = (http.get(url).raise_for_status().json() for url in (MODELS_URL, PROVIDERS_URL))
        return cls.of(models, providers)

    @classmethod
    def of(cls, models: dict[str, dict[str, Any]], providers: dict[str, dict[str, Any]]) -> Self:
        ours = json.loads(files(__package__).joinpath("models.json").read_text("utf-8"))
        return cls(models | ours, providers)

    def canonical(self, id: str) -> str:
        if id in self.models:
            return id
        listed = {f"{lab}/{id}" for lab, provider in self.providers.items() if id in provider.get("models", {}) and f"{lab}/{id}" in self.models}
        named = {entry["canonical_model_id"] for p in self.providers.values() if (entry := p["models"].get(id)) and "canonical_model_id" in entry}
        found = listed or named
        if len(found) != 1:
            raise KeyError(f"{id}: {'no' if not found else 'more than one'} models.dev model, {sorted(found)}")
        return found.pop()

    def hosting(self, id: str, provider: str) -> tuple[str, dict[str, Any]]:
        canonical = self.canonical(id)
        lab, _, local = canonical.partition("/")
        served = self.providers.get(provider, {}).get("models", {})
        hits = [(key, entry) for key, entry in served.items() if (provider == lab and key == local) or entry.get("canonical_model_id") == canonical]
        if not hits:
            raise KeyError(f"{provider} does not serve {canonical} on models.dev")
        return hits[0]

    def __getitem__(self, id: str) -> Model:
        canonical = self.canonical(id)
        lab, _, local = canonical.partition("/")
        entry, provider = self.models[canonical], self.providers.get(lab, {})
        listing = provider.get("models", {}).get(local)
        cost = (listing or {}).get("cost") or {}
        return Model(
            id=canonical,
            name=entry["name"],
            lab=lab,
            lab_name=provider.get("name", lab),
            released=entry["release_date"],
            deprecated=listing is None or listing.get("status") == DEPRECATED,
            price=Price(cost["input"], cost["output"]) if "input" in cost and "output" in cost else None,
        )
