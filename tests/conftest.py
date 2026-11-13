import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode

from geobench import GeoTaskset
from geobench.catalog import Catalog
from geobench.round import MAX_POINTS
from geobench.taskset import GeoTask, GeoTasksetConfig
from shungite import Coordinates, Guess, Item
from shungite.report import Median

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
STARTED = datetime(2025, 5, 22, 12, 52, 37, tzinfo=UTC).timestamp()


@dataclass(frozen=True)
class Graded:
    location: str
    truth: dict[str, Any]
    item: Item
    reply: str | None = None
    reasoning: str | None = None

    def trace(self, map: str, usage: dict[str, int] | None) -> dict[str, Any]:
        round = {"id": self.location, "country": self.truth["country"], "location": {"lat": self.truth["lat"], "lng": self.truth["lng"]}}
        return {
            "task": {"type": "GeoTask", "data": {"id": f"{map}/{self.location}", "round": round}},
            "nodes": [] if self.reply is None else [{"message": {"role": "assistant", "content": self.reply, "reasoning_content": self.reasoning}}],
            "info": {"items": {self.location: self.item.model_dump()}},
            "timing": {"start": STARTED},
            **({} if usage is None else {"usage": usage}),
        }


def answered(location: str, truth: dict[str, Any], country: str, lat: float, lng: float, points: float, km: float, passed: bool) -> Graded:
    guess = Guess(country=country, coordinates=Coordinates(lat=lat, lng=lng))
    return Graded(location, truth, Item(score=points / MAX_POINTS, passed=passed, guess=guess.model_dump(), refused=False, points=points, distance_km=km))


GRADED = [
    answered("1", IMAGES[0], "United States", 40.0, -90.0, 4233.0, 308.6, True),
    answered("2", IMAGES[1], "Tanzania", -3.0, 36.0, 4300.0, 285.1, False),
    answered("3", IMAGES[2], "Japan", 35.0, 139.0, 4700.0, 102.4, False),
    Graded("4", IMAGES[2], Item(score=0.0, passed=False, guess=None, refused=True, error_message="Format error: no answer")),
]


def write_run(
    dir: Path,
    graded: list[Graded] = GRADED,
    model: str = "lab/model-1",
    map: str = MAP,
    sampling: dict[str, Any] | None = None,
    median: Median = Median.HIGH,
    usage: dict[str, int] | None = None,
) -> Path:
    config = {"env": {"taskset": {"id": "geobench", "map": map}}, "model": model, "sampling": sampling, "report": {"median": median}}
    (dir / "configs" / "resolved").mkdir(parents=True)
    (dir / "configs" / "resolved" / "run.json").write_text(json.dumps(config))
    (dir / "traces.jsonl").write_text("".join(json.dumps({"traces": [g.trace(map, usage)]}) + "\n" for g in graded))
    return dir


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
