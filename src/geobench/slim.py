import argparse
import base64
import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from geobench.taskset import GeoTasksetConfig

INLINE_IMAGE = re.compile(rb"data:image/[a-z]+;base64,([A-Za-z0-9+/=]+)")
TRACES = "traces.jsonl"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Dataset:
    root: Path

    @cached_property
    def images(self) -> dict[str, Path]:
        return {sha256(path.read_bytes()): path.relative_to(self.root.parent) for path in self.root.rglob("images/*") if path.is_file()}

    def reference(self, inline: re.Match[bytes]) -> bytes:
        digest = sha256(base64.b64decode(inline[1]))
        return f"{self.images[digest].as_posix()}#sha256={digest}".encode()

    def slim(self, traces: bytes) -> bytes:
        return INLINE_IMAGE.sub(self.reference, traces)

    def slim_run(self, run: Path) -> None:
        traces = run / TRACES
        if not traces.is_file():
            return
        original = traces.read_bytes()
        slimmed = self.slim(original)
        if slimmed != original:
            staged = traces.with_suffix(".slim")
            staged.write_bytes(slimmed)
            staged.replace(traces)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m geobench.slim",
        description="Swap each image inlined in a run's traces for a reference to its dataset file.",
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=GeoTasksetConfig().dataset)
    args = parser.parse_args()
    dataset = Dataset(args.dataset)
    for run in args.runs:
        dataset.slim_run(run)


if __name__ == "__main__":
    main()
