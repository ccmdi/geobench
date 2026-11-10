from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Self

import httpx
from charoite import PanoRef, Panorama, Request

GAMES_URL = "https://www.geoguessr.com/api/v3/games"
SEARCH_RADIUS_M = 50.0


@dataclass(frozen=True)
class Location:
    lat: float
    lng: float
    pano: PanoRef | None
    heading: float
    pitch: float

    @classmethod
    def parse(cls, round: dict[str, Any]) -> Self:
        pano = round.get("panoId")
        return cls(
            lat=round["lat"],
            lng=round["lng"],
            pano=PanoRef.geoguessr(pano) if pano else None,
            heading=round.get("heading", 0.0),
            pitch=round.get("pitch", 0.0),
        )

    async def panorama(self, request: Request) -> Panorama:
        if self.pano:
            return await self.pano.panorama(request)
        return await Panorama.near(request, self.lat, self.lng, SEARCH_RADIUS_M)


@dataclass(frozen=True)
class GeoGuessr:
    http: httpx.AsyncClient
    ncfa: str

    async def call(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        response = await self.http.request(method, url, cookies={"_ncfa": self.ncfa}, **kwargs)
        response.raise_for_status()
        return response.json()

    async def locations(self, map_id: str) -> AsyncIterator[tuple[dict[str, Any], Location]]:
        settings = {"map": map_id, "type": "standard", "timeLimit": 0}
        settings |= {"forbidMoving": True, "forbidZooming": True, "forbidRotating": True}
        while True:
            game = await self.call("POST", GAMES_URL, json=settings)
            while True:
                yield game["bounds"], Location.parse(game["rounds"][game["round"] - 1])
                if game["round"] >= game["roundCount"]:
                    break
                url = f"{GAMES_URL}/{game['token']}"
                await self.call("POST", url, json={"token": game["token"], "lat": 0, "lng": 0, "timedOut": False, "stepsCount": 0})
                game = await self.call("GET", url)
