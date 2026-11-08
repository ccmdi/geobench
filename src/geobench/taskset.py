import json
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import verifiers.v1 as vf

from geobench.prompt import user_message
from geobench.round import MAX_POINTS, Round
from shungite import Bounds


class RoundData(vf.TaskData):
    round: Round


class GeoTask(vf.Task[RoundData, vf.State, vf.TaskConfig]):
    @vf.metric
    async def grade(self, trace: vf.Trace) -> dict[str, float]:
        return self.data.round.grade(trace.last_reply).metrics

    @vf.reward
    async def geoguessr(self, trace: vf.Trace) -> float:
        grade = self.data.round.grade(trace.last_reply)
        trace.info["verdict"] = grade.verdict
        return grade.points / MAX_POINTS


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
            image = PurePosixPath(raw["image_path"])
            yield GeoTask(
                RoundData(
                    id=f"{self.config.map}/{image.stem}",
                    prompt=[user_message(self.config.map_dir / image)],
                    round=Round.parse(raw, bounds),
                ),
                self.config.task,
            )
