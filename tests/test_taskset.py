import base64
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from conftest import IMAGES, MAP, image_bytes, taskset

from geobench.prompt import PROMPT
from geobench.round import Round
from geobench.taskset import GeoTask


def image_part(task: GeoTask) -> vf.ImageUrlContentPart:
    [message] = task.data.prompt
    [text, image] = message.content
    assert isinstance(text, vf.TextContentPart) and isinstance(image, vf.ImageUrlContentPart)
    return image


def test_every_image_becomes_a_task(tasks: dict[str, GeoTask]) -> None:
    assert list(tasks) == [f"{MAP}/1", f"{MAP}/2", f"{MAP}/3"]
    assert [task.data.round.country for task in tasks.values()] == [image["country"] for image in IMAGES]


def test_bounds_are_parsed_once_per_map(tasks: dict[str, GeoTask]) -> None:
    first, *rest = (task.data.round.bounds for task in tasks.values())
    assert all(bounds is first for bounds in rest)
    assert first.scale == pytest.approx(2015.3, abs=0.05)


def test_reference_location_is_typed_on_the_task(tasks: dict[str, GeoTask]) -> None:
    location = tasks[f"{MAP}/2"].data.round.location
    assert (location.lat, location.lng) == (IMAGES[1]["lat"], IMAGES[1]["lng"])


def test_prompt_is_the_instructions_and_the_image(tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/1"]
    assert task.data.system_prompt is None
    assert task.data.prompt_text == PROMPT
    assert "Only the last country/lat/lng block in your reply is graded." in PROMPT


@pytest.mark.parametrize("image", IMAGES, ids=lambda image: image["image_path"])
def test_image_is_sent_unmodified(tasks: dict[str, GeoTask], image: dict[str, Any]) -> None:
    url = image_part(tasks[f"{MAP}/{Path(image['image_path']).stem}"]).image_url.url
    header, encoded = url.split(",", 1)
    assert header == "data:image/jpeg;base64"
    assert base64.b64decode(encoded) == image_bytes(image["image_path"])


def test_no_reference_reaches_the_prompt(dataset: Path) -> None:
    text = "\n".join(task.data.prompt_text + image_part(task).image_url.url for task in taskset(dataset))
    for image in IMAGES:
        for value in (image["country"], str(image["lat"]), str(image["lng"])):
            assert value not in text


def test_rounds_round_trip_through_json(tasks: dict[str, GeoTask]) -> None:
    for task in tasks.values():
        assert Round.model_validate_json(task.data.round.model_dump_json()) == task.data.round


def test_missing_map_fails(dataset: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(taskset(dataset, "no-such-map"))
