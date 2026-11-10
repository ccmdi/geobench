import argparse
import json
from dataclasses import asdict
from pathlib import Path

from geobench.answers import answers

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an HTML map of every model's guess at each location.")
    parser.add_argument("root", nargs="?", type=Path, default=HERE.parents[1] / "responses", help="responses folder")
    parser.add_argument("output", nargs="?", type=Path, default=Path("visualization.html"), help="HTML file to write")
    args = parser.parse_args()
    data = json.dumps([asdict(answer) for answer in answers(args.root)]).replace("</", "<\\/")
    args.output.write_text((HERE / "template.html").read_text("utf-8").replace("__ANSWERS__", data), "utf-8")
    print(f"Visualization created: {args.output}")


if __name__ == "__main__":
    main()
