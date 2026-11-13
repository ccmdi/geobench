import json
import threading
from collections.abc import Iterator
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest
from conftest import CATALOG, GRADED, MAP, image_bytes, write_map, write_run

from geobench.leaderboard import Entry
from geobench.viz import Viewer, server

PUBLIC = f"lab_model-1_{MAP}_2026-11-13T03_32_04"
HIDDEN = f".hidden/lab_model-2_{MAP}_2026-11-12T22_00_00"


@pytest.fixture
def responses(tmp_path: Path) -> Path:
    root = tmp_path / "responses"
    write_run(root / PUBLIC, [replace(GRADED[0], reply="country: United States", reasoning="corn"), GRADED[3]], sampling={"temperature": 0.4})
    write_run(root / HIDDEN, GRADED[1:2], model="lab/model-2", sampling={"reasoning_effort": "high", "max_tokens": None}, client={"base_url": "https://api.lab.test/v1"})
    return root


@pytest.fixture
def base(tmp_path: Path, responses: Path) -> Iterator[str]:
    catalog: Future[Any] = Future()
    catalog.set_result(CATALOG)
    with server(Viewer(responses, write_map(tmp_path / "dataset"), catalog), 0) as http:
        threading.Thread(target=http.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{http.server_port}"
        http.shutdown()


def get(url: str) -> tuple[str, bytes]:
    with urlopen(url) as response:
        return response.headers["Content-Type"], response.read()


def api(base: str, path: str, **query: str) -> Any:
    return json.loads(get(f"{base}{path}?{urlencode(query)}")[1])


def status(url: str) -> int:
    with pytest.raises(HTTPError) as error:
        urlopen(url)
    return error.value.code


def test_the_run_list_reads_only_configs_and_sees_new_runs_without_a_restart(base: str, responses: Path) -> None:
    (responses / PUBLIC / "traces.jsonl").write_text("not json")
    listed = {run["run"]: run for run in api(base, "/api/runs")}
    assert set(listed) == {PUBLIC, HIDDEN}
    assert {key: listed[HIDDEN][key] for key in ("status", "name", "lab_name", "map", "effort", "thinking", "started", "provider", "settings")} == {
        "status": "hidden",
        "name": "Lab Model 2",
        "lab_name": "Lab",
        "map": MAP,
        "effort": "high",
        "thinking": True,
        "started": "2026-11-12T22:00:00+00:00",
        "provider": "api.lab.test",
        "settings": {"reasoning_effort": "high"},
    }
    assert listed[PUBLIC]["provider"] is None
    write_run(responses / ".incomplete" / f"lab_model-1_{MAP}_2026-11-14T00_00_00", [])
    assert len(api(base, "/api/runs")) == 3


def test_a_run_carries_the_leaderboards_summary_and_every_answer_with_its_reply(base: str, responses: Path) -> None:
    shown = api(base, "/api/run", run=PUBLIC)
    assert shown["summary"] == json.loads(json.dumps(Entry.read(responses, responses / PUBLIC).as_json(CATALOG)))
    assert shown["summary"]["n"] == 2
    usa, refused = shown["answers"]
    assert (usa["place"], usa["guess_place"], usa["passed"], usa["reply"], usa["reasoning"]) == ({"name": "United States", "code": "us"}, {"name": "United States", "code": "us"}, True, "country: United States", "corn")
    assert (refused["refused"], refused["points"], refused["reply"]) == (True, None, None)


@pytest.mark.parametrize("run", ["nope", "../responses", f"./{PUBLIC}", ".hidden"])
def test_only_listed_runs_load(base: str, run: str) -> None:
    assert status(f"{base}/api/run?{urlencode({'run': run})}") == 404


def test_a_locations_image_comes_from_its_map_in_the_dataset(base: str) -> None:
    assert get(f"{base}/api/image?{urlencode({'map': MAP, 'location': '2'})}") == ("image/jpeg", image_bytes("images/2.jpg"))


@pytest.mark.parametrize("query", [{"map": MAP, "location": "9"}, {"map": "..", "location": "1"}, {"map": "../responses", "location": "1"}, {"map": MAP}])
def test_images_outside_a_maps_rounds_are_refused(base: str, query: dict[str, str]) -> None:
    assert status(f"{base}/api/image?{urlencode(query)}") == 404


def test_the_page_and_its_modules_are_served_and_nothing_else(base: str) -> None:
    assert get(f"{base}/")[0] == "text/html"
    assert get(f"{base}/app.js")[0] == "text/javascript"
    assert status(f"{base}/../pyproject.toml") == 404
    assert status(f"{base}/static/app.js") == 404
