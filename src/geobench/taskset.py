import json
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from geobench.prompt import user_message
from geobench.round import Round
from shungite import Bounds


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


class GeoTaskset(vf.Taskset[GeoTask, GeoTasksetConfig]):
    def load(self) -> Iterator[GeoTask]:
        metadata = json.loads((self.config.map_dir / "metadata.json").read_text())
        bounds = Bounds.model_validate(metadata["bounds"])
        for raw in metadata["images"]:
            round = Round.parse(raw, bounds)
            yield GeoTask(
                RoundData(
                    id=f"{self.config.map}/{round.id}",
                    prompt=[user_message(self.config.map_dir / raw["image_path"])],
                    round=round,
                ),
                self.config.task,
            )
