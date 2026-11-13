import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from geobench.catalog import Catalog
from geobench.leaderboard import STARTED
from geobench.slim import Dataset
from geobench.taskset import GeoTasksetConfig

CONFIGS = Path("configs")
RESPONSES = Path("responses")
FALLBACK = "openrouter"
TEMPERATURE = 0.4


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geobench",
        description="Run a model on a GeoBench map into responses/, through its lab's API or else OpenRouter. Other options pass through to vf-run.",
    )
    parser.add_argument("model", help="model id, as its lab or models.dev names it")
    parser.add_argument("--map", default=GeoTasksetConfig().map, help="map folder in dataset/")
    parser.add_argument("--via", help="provider profile in configs/ to call through (default: the model's lab, else openrouter)")
    args, passthrough = parser.parse_known_args()

    load_dotenv(".env")
    catalog = Catalog.fetch()
    model = catalog[args.model]
    via = args.via or (model.lab if (CONFIGS / f"{model.lab}.toml").is_file() else FALLBACK)
    api_id, listing = catalog.hosting(model.id, via)
    sampling: list[str] = []
    if listing.get("temperature"):
        sampling += ["--sampling.temperature", str(TEMPERATURE)]
    else:
        print(f"{model.name} takes no temperature on {via}, so it runs at the provider default", file=sys.stderr)

    started = datetime.now(UTC).strftime(STARTED)
    run_dir = f"{api_id.replace('/', '_')}_{args.map}_{started}"
    vf_run = str(Path(sys.executable).parent / "vf-run")
    command = [
        vf_run,
        "@", str(CONFIGS / "base.toml"),
        "@", str(CONFIGS / f"{via}.toml"),
        "-m", api_id,
        "--env.taskset.map", args.map,
        "--select.shuffle", "True",
        "-o", str(RESPONSES),
        "--run.dir", run_dir,
        *sampling,
        *passthrough,
    ]
    print(f"{model.name} on {args.map} through {via} as {api_id}", file=sys.stderr)
    returncode = subprocess.run(command).returncode
    Dataset(GeoTasksetConfig().dataset).slim_run(RESPONSES / run_dir)
    sys.exit(returncode)


if __name__ == "__main__":
    main()
