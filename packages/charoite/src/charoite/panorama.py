import asyncio
import base64
import io
import math
import re
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any, Protocol, Self

import httpx
from PIL import Image

from charoite.schema import GetMetadataResponse, ImageKey, ImageMetadata, PanoType, SingleImageSearchResponse
from charoite.view import perspective

RPC_URL = "https://maps.googleapis.com/$rpc/google.internal.maps.mapsjs.v1.MapsJsInternalService"
OFFICIAL_TILE_URL = "https://streetviewpixels-pa.googleapis.com/v1/tile"
PHOTOSPHERE_TILE_URL = "https://lh3.ggpht.com/jsapi2/a/b/c/x{x}-y{y}-z{zoom}/{id}"
CLIENT = ["apiv3", None, None, None, "US", None, None, None, None, None, [[0]]]
LANGUAGE = ["en", "US"]
FIELDS = [[1, 2, 3, 4, 8, 6]]
OFFICIAL_ID = re.compile(r"[-_A-Za-z0-9]{21}[AQgw]")
NO_IMAGES = "Search returned no images."
USABLE_IMAGE = (1, 3)
TILE_PX = 512
TIMEOUT = 20


class Request(Protocol):
    def __call__(self, method: str, url: str, **kwargs: Any) -> Awaitable[httpx.Response]: ...


class StreetViewError(Exception):
    pass


@dataclass(frozen=True)
class PanoRef:
    type: PanoType
    id: str

    @classmethod
    def of(cls, key: ImageKey) -> Self:
        return cls(PanoType(key.frontend or PanoType.OFFICIAL), key.id)

    @classmethod
    def parse(cls, id: str) -> Self:
        if id.startswith("F:"):
            return cls(PanoType.UNKNOWN, id.removeprefix("F:"))
        if id.startswith("CIHM"):
            return cls(PanoType.USER_UPLOADED, id)
        if OFFICIAL_ID.fullmatch(id):
            return cls(PanoType.OFFICIAL, id)
        text = id.rstrip(".=")
        try:
            key = ImageKey.proto(base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_", validate=True))
        except (ValueError, IndexError):
            return cls(PanoType.OFFICIAL, id)
        ref = cls.of(key)
        return replace(ref, id=ref.id or id)

    @classmethod
    def geoguessr(cls, encoded: str) -> Self:
        try:
            id = bytes.fromhex(encoded).decode("ascii")
        except ValueError:
            id = encoded
        return cls.parse(id)

    def tile(self, x: int, y: int, zoom: int) -> tuple[str, dict[str, Any]]:
        match self.type:
            case PanoType.OFFICIAL:
                return OFFICIAL_TILE_URL, {"cb_client": "apiv3", "panoid": self.id, "x": x, "y": y, "zoom": zoom}
            case PanoType.USER_UPLOADED:
                return PHOTOSPHERE_TILE_URL.format(x=x, y=y, zoom=zoom, id=self.id), {}
            case PanoType.UNKNOWN:
                raise StreetViewError(f"No tile server is known for {self.type.name} panorama {self.id}.")

    async def panorama(self, request: Request) -> "Panorama":
        payload = [CLIENT, LANGUAGE, [[[self.type, self.id]]], FIELDS]
        image = await rpc(request, GetMetadataResponse, payload, missing=f"No Street View panorama {self.id}.")
        return Panorama.of(image)


@dataclass(frozen=True)
class Panorama:
    ref: PanoRef
    lat: float
    lng: float
    heading: float
    date: str | None
    levels: tuple[tuple[int, int], ...]

    @classmethod
    async def near(cls, request: Request, lat: float, lng: float, radius_m: float) -> Self:
        search = [
            CLIENT,
            [[None, None, lat, lng], radius_m],
            [None, LANGUAGE, None, None, None, None, None, None, [2], None, [[[2, True, 2]]]],
            FIELDS,
        ]
        missing = f"No Street View panorama within {radius_m:.0f} m of that point."
        return cls.of(await rpc(request, SingleImageSearchResponse, search, missing))

    @classmethod
    def of(cls, image: ImageMetadata) -> Self:
        [information, *_] = image.information
        date = image.date.date
        return cls(
            ref=PanoRef.of(image.pano),
            lat=information.location.location.lat,
            lng=information.location.location.lng,
            heading=information.location.pov.heading,
            date=f"{date.year:04d}-{date.month:02d}" if date.year else None,
            levels=tuple((level.size.width, level.size.height) for level in image.tiles.tile_size.possible),
        )

    def zoom(self, width: int) -> int:
        return next((z for z, (w, _) in enumerate(self.levels) if w >= width), len(self.levels) - 1)

    async def equirectangular(self, request: Request, width: int) -> "Equirectangular":
        zoom = self.zoom(width)
        size = self.levels[zoom]
        grid = [(x, y) for x in range(math.ceil(size[0] / TILE_PX)) for y in range(math.ceil(size[1] / TILE_PX))]
        tiles = await asyncio.gather(*(fetch(request, *self.ref.tile(x, y, zoom)) for x, y in grid))
        image = await asyncio.to_thread(stitch, size, zip(grid, tiles, strict=True))
        return Equirectangular(self, image)


@dataclass(frozen=True)
class Equirectangular:
    panorama: Panorama
    image: Image.Image

    def view(self, heading: float, pitch: float, fov: float, size: tuple[int, int]) -> Image.Image:
        return perspective(self.image, (heading - self.panorama.heading) % 360, pitch, fov, size)


async def rpc(
    request: Request,
    reply: type[SingleImageSearchResponse] | type[GetMetadataResponse],
    payload: list[Any],
    missing: str,
) -> ImageMetadata:
    method = reply.__name__.removesuffix("Response")
    response = await request(
        "POST",
        f"{RPC_URL}/{method}",
        json=payload,
        headers={"Content-Type": "application/json+protobuf"},
        timeout=TIMEOUT,
    )
    answer = reply(response.json())
    if answer.status.error_message == NO_IMAGES:
        raise StreetViewError(missing)
    if answer.status.code:
        raise StreetViewError(f"Street View {method} failed: {answer.status.error_message or answer.status.code}")
    if answer.image.status.code not in USABLE_IMAGE:
        raise StreetViewError(missing)
    return answer.image


async def fetch(request: Request, url: str, params: dict[str, Any]) -> bytes:
    return (await request("GET", url, params=params, timeout=TIMEOUT)).content


def stitch(size: tuple[int, int], tiles: Any) -> Image.Image:
    canvas = Image.new("RGB", size)
    for (x, y), tile in tiles:
        canvas.paste(Image.open(io.BytesIO(tile)), (x * TILE_PX, y * TILE_PX))
    return canvas
