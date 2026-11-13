import base64
import hashlib
from pathlib import Path

import pytest
from conftest import MAP

from geobench.slim import Dataset
from geobench.taskset import GeoTask, RoundData


def image_url(data: RoundData) -> str:
    [message] = data.prompt
    return message.content[1].image_url.url


def test_a_slimmed_task_points_at_the_exact_dataset_image(dataset: Path, tasks: dict[str, GeoTask]) -> None:
    task = tasks[f"{MAP}/1"]
    inline = image_url(task.data)

    slimmed = RoundData.model_validate_json(Dataset(dataset).slim(task.data.model_dump_json().encode()))

    path, digest = image_url(slimmed).split("#sha256=")
    original = base64.b64decode(inline.split(",", 1)[1])
    assert (dataset.parent / path).read_bytes() == original
    assert digest == hashlib.sha256(original).hexdigest()
    assert slimmed.model_dump(exclude={"prompt"}) == task.data.model_dump(exclude={"prompt"})


def test_an_image_missing_from_the_dataset_is_refused(dataset: Path) -> None:
    stranger = b"data:image/jpeg;base64," + base64.b64encode(b"not in the dataset")
    with pytest.raises(KeyError):
        Dataset(dataset).slim(stranger)
