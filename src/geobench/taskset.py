import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from geobench.prompt import user_message
from geobench.round import Bounds, Round


class RoundData(vf.TaskData):
    round: Round


class GeoTask(vf.Task[RoundData, vf.State, vf.TaskConfig]):
    @vf.metric
    async def grade(self, trace: vf.Trace) -> dict[str, float]:
        return self.data.round.grade(trace.last_reply).metrics

    @vf.reward
    async def geoguessr(self, trace: vf.Trace) -> float:
        item = self.data.round.grade(trace.last_reply).item
        trace.info["items"] = {self.data.round.id: item.model_dump()}
        return item.score


class GeoTasksetConfig(vf.TasksetConfig):
    dataset: Path = Path("dataset")
    map: str = "acw-02-25-25"
    task: vf.TaskConfig = vf.TaskConfig()

    @property
    def map_dir(self) -> Path:
        return self.dataset / self.map

    @property
    def rounds(self) -> Iterator[tuple[Round, Path]]:
        metadata = json.loads((self.map_dir / "metadata.json").read_text())
        bounds = Bounds.model_validate(metadata["bounds"])
        return ((Round.parse(raw, bounds), self.map_dir / raw["image_path"]) for raw in metadata["images"])


class GeoTaskset(vf.Taskset[GeoTask, GeoTasksetConfig]):
    def load(self) -> Iterator[GeoTask]:
        for round, image in self.config.rounds:
            yield GeoTask(RoundData(id=f"{self.config.map}/{round.id}", prompt=[user_message(image)], round=round), self.config.task)
