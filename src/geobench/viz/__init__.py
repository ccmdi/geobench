import argparse
import json
import mimetypes
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

from geobench.answers import Answer
from geobench.catalog import Catalog
from geobench.leaderboard import Entry, run_dirs
from geobench.taskset import GeoTasksetConfig

STATIC = Path(__file__).parent / "static"
PORT = 8350


class Missing(LookupError):
    pass


class Query(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise Missing(f"no {key} given")


@dataclass(frozen=True)
class Viewer:
    responses: Path
    dataset: Path
    catalog: Future[Catalog]

    def entry(self, run: str) -> Entry:
        dirs = {dir.relative_to(self.responses).as_posix(): dir for dir in run_dirs(self.responses)}
        if run not in dirs:
            raise Missing(f"no run {run}")
        return Entry.read(self.responses, dirs[run])

    def runs(self) -> list[dict[str, Any]]:
        catalog = self.catalog.result()
        return [
            entry.card(catalog) | {"started": entry.started, "provider": entry.provider, "settings": entry.settings}
            for entry in (Entry.read(self.responses, dir) for dir in run_dirs(self.responses))
        ]

    def run(self, run: str) -> dict[str, Any]:
        entry = self.entry(run)
        return {"summary": entry.as_json(self.catalog.result()), "answers": [asdict(answer) for answer in Answer.each(entry)]}

    def image(self, map: str, location: str) -> Path:
        maps = {dir.name for dir in self.dataset.iterdir() if dir.is_dir()}
        if map not in maps:
            raise Missing(f"no map {map}")
        found = [image for round, image in GeoTasksetConfig(dataset=self.dataset, map=map).rounds if round.id == location]
        if not found or not found[0].resolve().is_relative_to(self.dataset.resolve()):
            raise Missing(f"no image for {map}/{location}")
        return found[0]


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args: Any, viewer: Viewer, **kwargs: Any) -> None:
        self.viewer = viewer
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = Query(parse_qsl(url.query))
        try:
            match url.path:
                case "/api/runs":
                    self.json(self.viewer.runs())
                case "/api/run":
                    self.json(self.viewer.run(query["run"]))
                case "/api/image":
                    self.file(self.viewer.image(query["map"], query["location"]))
                case "/":
                    self.file(STATIC / "index.html")
                case _:
                    self.file(static(url.path))
        except Missing as missing:
            self.send_error(HTTPStatus.NOT_FOUND, str(missing))

    def json(self, value: Any) -> None:
        self.send(json.dumps(value, ensure_ascii=False).encode(), "application/json")

    def file(self, path: Path) -> None:
        self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def send(self, body: bytes, type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def static(path: str) -> Path:
    files = {f"/{file.name}": file for file in STATIC.iterdir()}
    if path not in files:
        raise Missing(f"no file {path}")
    return files[path]


def server(viewer: Viewer, port: int = PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, viewer=viewer))


def main() -> None:
    parser = argparse.ArgumentParser(prog="viz", description="Browse and compare runs' guesses in the browser.")
    parser.add_argument("--responses", type=Path, default=Path("responses"), help="responses folder")
    parser.add_argument("--dataset", type=Path, default=GeoTasksetConfig().dataset, help="dataset folder")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    with ThreadPoolExecutor(1) as pool, server(Viewer(args.responses, args.dataset, pool.submit(Catalog.fetch)), args.port) as http:
        url = f"http://127.0.0.1:{http.server_port}/"
        print(url)
        webbrowser.open(url)
        with suppress(KeyboardInterrupt):
            http.serve_forever()
