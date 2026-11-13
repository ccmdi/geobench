import json
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode

from geobench import GeoTaskset
from geobench.catalog import Catalog
from geobench.taskset import GeoTask, GeoTasksetConfig
from shungite import Coordinates

ROOT = Path(__file__).parents[1]

MAP = "test-map"
CATALOG = Catalog.of(
    {
        "lab/model-1": {"id": "lab/model-1", "name": "Lab Model 1", "release_date": "2025-01-14"},
        "lab/model-2": {"id": "lab/model-2", "name": "Lab Model 2", "release_date": "2024-06"},
    },
    {"lab": {"id": "lab", "name": "Lab", "models": {"model-1": {"id": "model-1", "cost": {"input": 2, "output": 8}}}}},
)
BOUNDS = {"min": {"lat": -55.0, "lng": -170.0}, "max": {"lat": 80.0, "lng": 178.0}}
IMAGES: list[dict[str, Any]] = [
    {"image_path": "images/1.jpg", "country": "United States of America (the)", "lat": 39.4817, "lng": -98.2263},
    {"image_path": "images/2.jpg", "country": "Kenya", "lat": 0.6142, "lng": 37.3519},
    {"image_path": "images/3.jpg", "country": "Japan", "lat": 36.1374, "lng": 138.4409},
]


def image_bytes(name: str) -> bytes:
    return b"\xff\xd8\xff\xe0" + name.encode()


def write_map(root: Path, images: list[dict[str, Any]] = IMAGES) -> Path:
    map_dir = root / MAP
    (map_dir / "images").mkdir(parents=True)
    (map_dir / "metadata.json").write_text(json.dumps({"bounds": BOUNDS, "images": images}))
    for image in images:
        (map_dir / image["image_path"]).write_bytes(image_bytes(image["image_path"]))
    return root


def taskset(dataset: Path, map: str = MAP) -> GeoTaskset:
    return GeoTaskset(GeoTasksetConfig(id="geobench", dataset=dataset, map=map))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return write_map(tmp_path)


@pytest.fixture
def tasks(dataset: Path) -> dict[str, GeoTask]:
    return {task.data.id: task for task in taskset(dataset)}


def replied(task: GeoTask, reply: str) -> vf.Trace:
    return vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="GeoTask", data=task.data),
        nodes=[
            MessageNode(parent=None, message=vf.UserMessage(content="image"), sampled=False),
            MessageNode(parent=0, message=vf.AssistantMessage(content=reply), sampled=True),
        ],
    )


def final(country: str, point: Coordinates) -> str:
    return f"My reasoning.\n\ncountry: {country}\nlat: {point.lat}\nlng: {point.lng}"
