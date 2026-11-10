import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from geobench.answers import answers
from shungite.country import canonical_country


def by_country(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    countries = frame.assign(country=frame["country"].map(lambda name: canonical_country(name) or name))
    points = countries.groupby(["country", "model"])["points"]
    return points.mean().unstack(), points.size().unstack().astype("Int64")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write each model's average score and sample count per country to CSV.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).parents[1] / "responses", help="responses folder")
    args = parser.parse_args()
    means, counts = by_country(pd.DataFrame(asdict(answer) for answer in answers(args.root)))
    combined = (means.map("{:.1f}".format) + " (n=" + counts.astype(str) + ")").where(means.notna())
    for title, table, file in (
        ("Average Country Scores by Model:", means, "country_avg_scores_by_model.csv"),
        ("Sample Counts by Country and Model:", counts, "country_sample_counts_by_model.csv"),
        ("Combined Scores with Sample Counts:", combined, "country_scores_with_counts_by_model.csv"),
    ):
        print(title, table, "", sep="\n")
        table.to_csv(file)


if __name__ == "__main__":
    main()
