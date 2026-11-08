import pytest

from shungite import Coordinates, Guess, read_guess


def guess(country: str | None, lat: float, lng: float) -> Guess:
    return Guess(country=country, coordinates=Coordinates(lat=lat, lng=lng))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("country: Japan\nlat: 35.6\nlng: 139.7", guess("Japan", 35.6, 139.7)),
        ("country: Réunion, France\nlat: -21.128\nlng: 55.454", guess("Réunion, France", -21.128, 55.454)),
        ("country: France (Réunion)\nlat: -21.1\nlng: 55.5", guess("France (Réunion)", -21.1, 55.5)),
        ("Country: Chile\nLat: -33.4\nLng: -70.6", guess("Chile", -33.4, -70.6)),
        ("COUNTRY: Chile\nLATITUDE: -33.4\nLONGITUDE: -70.6", guess("Chile", -33.4, -70.6)),
        ("country: Chile\nlatitude: -33.4\nlon: -70.6", guess("Chile", -33.4, -70.6)),
        ("country: Chile\nlat: -33.4\nlong: -70.6", guess("Chile", -33.4, -70.6)),
        ("**country:** Peru\n**lat:** -12.0\n**lng:** -77.0", guess("Peru", -12.0, -77.0)),
        ("**country: Peru**\n**lat: -12.0**\n**lng: -77.0**", guess("Peru", -12.0, -77.0)),
        ("- **Country**: Peru\n- **Latitude**: -12.0\n- **Longitude**: -77.0", guess("Peru", -12.0, -77.0)),
        ("*   country: New Zealand\n*   lat: -43.0\n*   lng: 172.5", guess("New Zealand", -43.0, 172.5)),
        ("country: _Peru_\nlat: `-12.0`\nlng: *-77.0*", guess("Peru", -12.0, -77.0)),
        ('"country": "Peru",\n"lat": -12.0,\n"lng": -77.0', guess("Peru", -12.0, -77.0)),
        ("country: Peru.\nlat: -12.0\nlng: -77.0", guess("Peru", -12.0, -77.0)),
    ],
)
def test_labelled_lines(reply: str, expected: Guess) -> None:
    assert read_guess(reply) == expected


@pytest.mark.parametrize(
    ("lat", "lng", "expected"),
    [
        ("10.5 N", "75.0 W", (10.5, -75.0)),
        ("46.5°N", "7.5°E", (46.5, 7.5)),
        ("46.5° S", "7.5° W", (-46.5, -7.5)),
        ("-21.1°", "55.4°", (-21.1, 55.4)),
        ("12.5 north", "70 west", (12.5, -70.0)),
        ("approximately 12.5 degrees South", "about 70 degrees East", (-12.5, 70.0)),
        ("~46.5°N", "~7.5°E", (46.5, 7.5)),
        ("46.5n", "7.5w", (46.5, -7.5)),
        ("-46.5 S", "-75.0° W", (-46.5, -75.0)),
        ("+12", "+45", (12.0, 45.0)),
        (".5", "-.25", (0.5, -0.25)),
        ("12.", "45.", (12.0, 45.0)),
        ("90", "-180", (90.0, -180.0)),
        ("9.7489° N", "84.0370° W", (9.7489, -84.037)),
    ],
)
def test_number_forms(lat: str, lng: str, expected: tuple[float, float]) -> None:
    assert read_guess(f"lat: {lat}\nlng: {lng}") == guess(None, *expected)


@pytest.mark.parametrize(
    "lat",
    [
        "+46.5 S",
        "-46.5 N",
        "46.5 E",
        "91",
        "12,5",
        "41° 44' 44.45\" N",
        "Unable to estimate",
        "Given the vegetation, around 0° plus or minus a few degrees",
        "46.5km",
    ],
)
def test_unreadable_latitude_is_no_value(lat: str) -> None:
    assert read_guess(f"lat: {lat}\nlng: 10.0") is None


def test_prose_after_key_is_no_value() -> None:
    reply = "Longitude: Since longitude spans from -180° to +180°, it cannot be estimated.\nlat: 0°\nlng: Unable to estimate"
    assert read_guess("lat: 1.0\n" + reply) is None


def test_longitude_hemisphere_must_be_east_or_west() -> None:
    assert read_guess("lat: 10.0\nlng: 75.0 N") is None
    assert read_guess("lat: 10.0\nlng: 181") is None


@pytest.mark.parametrize(
    "line",
    [
        "lng: 99.99</thinking>",
        "lng: 99.99 (This is approximate for West Bengal region)",
        "lng: 99.99°E (99.99)",
        "lng: 99.99 I'm leaning towards Uganda based on the landscape.",
    ],
)
def test_trailing_text_after_value(line: str) -> None:
    assert read_guess(f"country: India\nlat: 22.5\n{line}") == guess("India", 22.5, 99.99)


def test_digit_typo_before_key() -> None:
    assert read_guess("country: Japan\nlat: 28.2714\n0lng: 129.3352") == guess("Japan", 28.2714, 129.3352)


def test_numbered_list_prefix() -> None:
    assert read_guess("1. country: Japan\n2. lat: 35.0\n3. lng: 139.0") == guess("Japan", 35.0, 139.0)


def test_country_is_optional() -> None:
    assert read_guess("lat: 41.74568\nlng: -70.58822") == guess(None, 41.74568, -70.58822)


def test_crlf_line_endings() -> None:
    assert read_guess("country: Chile\r\nlat: -33.4\r\nlng: -70.6\r\n") == guess("Chile", -33.4, -70.6)


def test_empty_country_line_is_no_country() -> None:
    assert read_guess("country:\nlat: 1.0\nlng: 2.0") == guess(None, 1.0, 2.0)


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I'm sorry, but I can't help with that.",
        "country: Japan",
        "country: Japan\nlat: 35.0",
        "lat: Unable to estimate accurately\nlng: Unable to estimate accurately",
        "The coordinates are 35.0, 139.0.",
        "lat/lng: 35.0, 139.0",
        "lat: 35.0, lng: 139.0",
        "lat: " + "9" * 500 + "\nlng: 1",
    ],
)
def test_no_complete_pair(reply: str) -> None:
    assert read_guess(reply) is None


def test_last_complete_block_wins_over_thinking_draft() -> None:
    reply = (
        "<thinking>Let me look.\n"
        "country: Thailand\nlat: 17.5\nlng: 102.8\n"
        "That fits Isan.</thinking>\n\n"
        "Final answer:\n"
        "country: Thailand\nlat: 17.0\nlng: 103.0"
    )
    assert read_guess(reply) == guess("Thailand", 17.0, 103.0)


def test_last_block_wins_over_restated_answer() -> None:
    reply = "country: Ghana\nlat: 6.5\nlng: -1.0\n\nOn reflection:\ncountry: Uganda\nlat: 0.8\nlng: 31.5"
    assert read_guess(reply) == guess("Uganda", 0.8, 31.5)


def test_blocks_do_not_splice_across_a_malformed_line() -> None:
    reply = (
        "country: Laos/Cambodia\nlat: 15°N\nlng: 105°E\n"
        "country: Laos\nlat: 16.2\nlng: 105.8"
    )
    assert read_guess(reply) == guess("Laos", 16.2, 105.8)


def test_reasoning_lines_before_answer_do_not_leak_country() -> None:
    reply = "Latitude: 37.5665° N\nLongitude: 126.9780° E\n\ncountry: South Korea\nlat: 37.51\nlng: 127.10"
    assert read_guess(reply) == guess("South Korea", 37.51, 127.10)


def test_country_starts_a_new_block() -> None:
    assert read_guess("lat: 1.0\ncountry: Kenya\nlng: 36.8") is None


def test_repeated_axis_restarts_block() -> None:
    assert read_guess("country: Kenya\nlat: 1.0\nlat: -1.3\nlng: 36.8") == guess(None, -1.3, 36.8)


def test_coordinates_are_strict() -> None:
    with pytest.raises(ValueError):
        Coordinates(lat="1.0", lng=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Coordinates(lat=90.5, lng=0.0)
    with pytest.raises(ValueError):
        Coordinates(lat=0.0, lng=-180.5)


def test_guess_is_frozen() -> None:
    with pytest.raises(ValueError):
        guess("Japan", 1.0, 2.0).country = "Chile"  # type: ignore[misc]
