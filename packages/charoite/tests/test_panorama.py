import io
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image

from charoite import Equirectangular, PanoRef, PanoType, Panorama, StreetViewError
from charoite.schema import GetMetadataResponse, ImageKey, ImageMetadata, Message, SingleImageSearchResponse

HERE = Path(__file__).parent
SEARCH = json.loads((HERE / "streetview_search.json").read_text())
METADATA = json.loads((HERE / "streetview_metadata.json").read_text())
EIFFEL = PanoRef(PanoType.OFFICIAL, "Xta4ugN_QRTIo3XDFLujgw")
SEARCHED = Panorama.of(SingleImageSearchResponse(SEARCH).image)
SEARCH_BODY = (
    '[["apiv3",null,null,null,"US",null,null,null,null,null,[[0]]],[[null,null,48.8584,2.2945],50.0],'
    '[null,["en","US"],null,null,null,null,null,null,[2],null,[[[2,true,2]]]],[[1,2,3,4,8,6]]]'
)
METADATA_BODY = (
    '[["apiv3",null,null,null,"US",null,null,null,null,null,[[0]]],["en","US"],'
    '[[[2,"Xta4ugN_QRTIo3XDFLujgw"]]],[[1,2,3,4,8,6]]]'
)
DEAD_PANO = [[0], [[[2], [2, "AAAAAAAAAAAAAAAAAAAAAA"], None, None, None, None, None, []]]]

Handler = Callable[[httpx.Request], httpx.Response]


def jpeg(size: tuple[int, int], color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def rpc_handler(reply: object, seen: list[httpx.Request]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "$rpc" in request.url.path:
            return httpx.Response(200, json=reply)
        return httpx.Response(200, content=jpeg((512, 512), "gray"))

    return handler


def test_geoguessr_hex_id_decodes_to_an_official_panorama() -> None:
    assert PanoRef.geoguessr(EIFFEL.id.encode().hex()) == EIFFEL


def test_geoguessr_id_that_is_not_hex_is_taken_as_is() -> None:
    assert PanoRef.geoguessr(EIFFEL.id) == EIFFEL


@pytest.mark.parametrize(
    ("id", "ref"),
    [
        ("-zrYsLR4Fh-cfJG_EMZ1-A", PanoRef(PanoType.OFFICIAL, "-zrYsLR4Fh-cfJG_EMZ1-A")),
        ("CAoSF0NJSE0wb2dLRUlDQWdJQ0VtX2l4cXdF", PanoRef(PanoType.USER_UPLOADED, "CIHM0ogKEICAgICEm_ixqwE")),
        ("CAoSA2FiYw..", PanoRef(PanoType.USER_UPLOADED, "abc")),
        ("CIHM0ogKEICAgICTzu7WYg", PanoRef(PanoType.USER_UPLOADED, "CIHM0ogKEICAgICTzu7WYg")),
        ("F:abc123", PanoRef(PanoType.UNKNOWN, "abc123")),
        ("not base64 at all", PanoRef(PanoType.OFFICIAL, "not base64 at all")),
        ("", PanoRef(PanoType.OFFICIAL, "")),
    ],
)
def test_pano_ids_resolve_to_their_image_key(id: str, ref: PanoRef) -> None:
    assert PanoRef.parse(id) == ref
    assert PanoRef.geoguessr(id.encode().hex()) == ref


def test_a_key_longer_than_one_length_byte_keeps_its_whole_id() -> None:
    key = "CAoSyAF4" + "eHh4" * 66 + "eA.."
    assert PanoRef.parse(key) == PanoRef(PanoType.USER_UPLOADED, "x" * 200)


def test_tiles_route_by_kind() -> None:
    assert EIFFEL.tile(1, 2, 3) == (
        "https://streetviewpixels-pa.googleapis.com/v1/tile",
        {"cb_client": "apiv3", "panoid": EIFFEL.id, "x": 1, "y": 2, "zoom": 3},
    )
    assert PanoRef(PanoType.USER_UPLOADED, "CIHM").tile(1, 2, 3) == ("https://lh3.ggpht.com/jsapi2/a/b/c/x1-y2-z3/CIHM", {})


def test_unknown_imagery_has_no_tile_server() -> None:
    with pytest.raises(StreetViewError, match="No tile server"):
        PanoRef(PanoType.UNKNOWN, "abc").tile(0, 0, 0)


def test_search_and_metadata_replies_parse_to_the_same_panorama() -> None:
    assert SEARCHED == Panorama.of(GetMetadataResponse(METADATA).image)
    assert SEARCHED.ref == EIFFEL
    assert (SEARCHED.lat, SEARCHED.lng) == pytest.approx((48.858440, 2.294514))
    assert SEARCHED.heading == pytest.approx(314.63293)
    assert SEARCHED.date == "2013-06"
    assert SEARCHED.levels[3] == (3328, 1664)


def test_an_image_without_a_capture_date_has_none() -> None:
    assert Panorama.of(ImageMetadata(SEARCH[1][:6])).date is None


def test_a_view_reads_only_fields_its_proto_declares() -> None:
    with pytest.raises(KeyError):
        type("ImageKey", (Message,), {"__annotations__": {"panoid": str}})
    with pytest.raises(TypeError):
        type("ImageKey", (Message,), {"__annotations__": {"id": int}})
    with pytest.raises(TypeError):
        type("ImageTileSize", (Message,), {"__annotations__": {"possible": ImageKey}})


@pytest.mark.parametrize(("width", "zoom"), [(1, 0), (2048, 3), (3328, 3), (3329, 4), (99999, 5)])
def test_zoom_is_the_smallest_level_at_least_that_wide(width: int, zoom: int) -> None:
    assert SEARCHED.zoom(width) == zoom


async def test_near_searches_and_parses() -> None:
    seen: list[httpx.Request] = []
    async with client(rpc_handler(SEARCH, seen)) as http:
        pano = await Panorama.near(http.request, 48.8584, 2.2945, 50.0)
    assert pano.ref == EIFFEL
    [request] = seen
    assert request.url.path.endswith("/SingleImageSearch")
    assert request.content.decode() == SEARCH_BODY


async def test_lookup_by_ref_reads_metadata() -> None:
    seen: list[httpx.Request] = []
    async with client(rpc_handler(METADATA, seen)) as http:
        pano = await EIFFEL.panorama(http.request)
    assert pano == SEARCHED
    assert seen[0].url.path.endswith("/GetMetadata")
    assert seen[0].content.decode() == METADATA_BODY


async def test_a_dead_pano_is_missing() -> None:
    async with client(rpc_handler(DEAD_PANO, [])) as http:
        with pytest.raises(StreetViewError, match="No Street View panorama AAAA"):
            await PanoRef.parse("AAAAAAAAAAAAAAAAAAAAAA").panorama(http.request)


async def test_no_coverage_names_the_radius() -> None:
    async with client(rpc_handler([[5, "generic", "Search returned no images."]], [])) as http:
        with pytest.raises(StreetViewError, match="No Street View panorama within 50 m"):
            await Panorama.near(http.request, 0.0, -150.0, 50.0)


async def test_unexpected_status_is_an_error() -> None:
    async with client(rpc_handler([[5, "generic", "decommissioned"]], [])) as http:
        with pytest.raises(StreetViewError, match="SingleImageSearch failed.*decommissioned"):
            await Panorama.near(http.request, 1.0, 1.0, 50.0)


async def test_equirectangular_stitches_every_tile_of_the_chosen_level() -> None:
    seen: list[httpx.Request] = []
    async with client(rpc_handler(SEARCH, seen)) as http:
        sphere = await SEARCHED.equirectangular(http.request, 2048)
    assert sphere.image.size == (3328, 1664)
    assert len(seen) == 7 * 4
    assert {r.url.params["zoom"] for r in seen} == {"3"}


async def test_photosphere_tiles_come_from_lh3() -> None:
    seen: list[httpx.Request] = []
    pano = Panorama(PanoRef(PanoType.USER_UPLOADED, "CIHM"), 0.0, 0.0, 0.0, None, ((1024, 512),))
    async with client(rpc_handler(None, seen)) as http:
        await pano.equirectangular(http.request, 1024)
    assert {r.url.host for r in seen} == {"lh3.ggpht.com"}
    assert len(seen) == 2


def banded(heading: float) -> Equirectangular:
    image = Image.new("RGB", (720, 360), "blue")
    image.paste(Image.new("RGB", (20, 360), "red"), (350, 0))
    return Equirectangular(Panorama(EIFFEL, 0.0, 0.0, heading, None, ((720, 360),)), image)


@pytest.mark.parametrize("heading", [0.0, 90.0, 314.6])
def test_view_faces_the_compass_heading(heading: float) -> None:
    sphere = banded(heading)
    ahead = sphere.view(heading, 0.0, 30.0, (64, 36)).getpixel((32, 18))
    behind = sphere.view(heading + 180, 0.0, 30.0, (64, 36)).getpixel((32, 18))
    assert ahead[0] > 200 and ahead[2] < 60
    assert behind[2] > 200 and behind[0] < 60


def test_view_has_the_requested_size() -> None:
    assert banded(0.0).view(0.0, 10.0, 110.0, (192, 108)).size == (192, 108)


@pytest.mark.live
async def test_live_search_lookup_and_view() -> None:
    async with httpx.AsyncClient(timeout=30) as http:
        found = await Panorama.near(http.request, 48.8584, 2.2945, 50.0)
        assert await found.ref.panorama(http.request) == found
        sphere = await found.equirectangular(http.request, 2048)
    assert sphere.image.size == found.levels[found.zoom(2048)]
    assert sphere.view(found.heading, 0.0, 110.0, (192, 108)).size == (192, 108)
