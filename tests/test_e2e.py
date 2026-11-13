import asyncio
import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest
import verifiers.v1 as vf
from conftest import IMAGES, MAP, ROOT, final, image_bytes, write_map
from shungite.testing import FakeModel, Reply
from verifiers.v1.utils.trace_store import read_episodes

from geobench.prompt import PROMPT
from geobench.taskset import RoundData
from shungite import Coordinates

needs_tmp = pytest.mark.skipif(not os.access("/tmp", os.W_OK), reason="verifiers writes /tmp/vf-port-*")


async def vf_run(model: FakeModel, dataset: Path, out: Path, *options: str) -> None:
    async with model.serving() as base_url:
        process = await asyncio.create_subprocess_exec(
            str(Path(sys.executable).parent / "vf-run"),
            "@", "configs/base.toml",
            "-m", "fake-model",
            "--client.base-url", base_url,
            "--client.api-key-var", "GEOBENCH_FAKE_KEY",
            "--env.taskset.dataset", str(dataset),
            "--env.taskset.map", MAP,
            "-o", str(out),
            "--run.dir", "e2e",
            *options,
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
    model = FakeModel([Reply(final("United States", reference))])
    await vf_run(model, write_map(tmp_path / "dataset"), tmp_path / "out", "-n", "1")

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
    assert trace.info["items"]["1"]["guess"]["country"] == "United States"
    assert trace.info["items"]["1"]["passed"] is True


@needs_tmp
async def test_two_trials_per_round_report_pass_at_k_and_pass_hat_k(tmp_path: Path) -> None:
    first, second = (Coordinates(lat=image["lat"], lng=image["lng"]) for image in IMAGES[:2])
    model = FakeModel([Reply(final("United States", first)), Reply(final("Canada", first)), Reply(final("Kenya", second)), Reply("I cannot tell.")])
    await vf_run(model, write_map(tmp_path / "dataset"), tmp_path / "out", "-n", "2", "-r", "2", "-c", "1")
    assert all(not any(m["role"] == "assistant" for m in request["messages"]) for request in model.requests)

    report = subprocess.run(
        [sys.executable, "-m", "shungite.report", str(tmp_path / "out" / "e2e"), "--price-in", "1000000", "--price-out", "2000000"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    *row, _date = report[4].split(" | ")
    assert row == ["fake-model (reasoning_effort high)", "2", "50.0%", "0.0%", "$6.00", "0.750 (0.500-1.000)", "100.0%", "2"]
    assert f"{MAP}/1 | 1 | - | 1, 1 | 1/2" in report
    assert f"{MAP}/2 | 2 | - | 1, 0 | 1/2" in report
    assert "Model input tokens | 4 | 1.0" in report
