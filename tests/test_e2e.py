import asyncio
import base64
import os
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf
from conftest import IMAGES, MAP, ROOT, final, image_bytes, write_map
from fake_model import FakeModel
from verifiers.v1.utils.trace_store import read_episodes

from geobench.prompt import PROMPT
from geobench.taskset import RoundData
from shungite import Coordinates

needs_tmp = pytest.mark.skipif(not os.access("/tmp", os.W_OK), reason="verifiers writes /tmp/vf-port-*")


async def vf_run(model: FakeModel, dataset: Path, out: Path) -> None:
    async with model.serving() as base_url:
        process = await asyncio.create_subprocess_exec(
            str(Path(sys.executable).parent / "vf-run"),
            "@", "configs/base.toml",
            "-m", "fake-model",
            "--client.base-url", base_url,
            "--client.api-key-var", "GEOBENCH_FAKE_KEY",
            "--env.taskset.dataset", str(dataset),
            "--env.taskset.map", MAP,
            "-n", "1",
            "-o", str(out),
            "--run.dir", "e2e",
            cwd=ROOT,
            env=os.environ | {"GEOBENCH_FAKE_KEY": "fake"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=600)
    assert process.returncode == 0, output.decode()


@needs_tmp
async def test_vf_run_scores_the_reference_location_full_points(tmp_path: Path) -> None:
    [first, *_] = IMAGES
    reference = Coordinates(lat=first["lat"], lng=first["lng"])
    model = FakeModel([final("United States", reference)])
    await vf_run(model, write_map(tmp_path / "dataset"), tmp_path / "out")

    [request] = model.requests
    assert "tools" not in request
    [user] = [m for m in request["messages"] if m["role"] == "user"]
    text, image = user["content"]
    assert text["text"] == PROMPT
    assert image["image_url"]["url"] == "data:image/jpeg;base64," + base64.b64encode(image_bytes(first["image_path"])).decode()

    [episode] = read_episodes(tmp_path / "out" / "e2e", vf.Trace[RoundData])
    [trace] = episode.traces
    assert trace.ok
    assert not trace.is_truncated
    assert trace.reward == 1.0
    assert trace.metrics == {"score": 5000.0, "country_correct": 1.0, "answered": 1.0, "distance_km": 0.0}
    assert trace.info["verdict"]["guess"]["country"] == "United States"
