from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from geobench.answers import answers

RESPONSES = Path(__file__).parents[2] / "responses"
ALL_TESTS = "All Tests"


@dataclass(frozen=True)
class Metric:
    column: str
    name: str
    label: str

    def by_country(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(["model", "country"])[self.column].mean().astype(float).rename(self.name).reset_index()


METRICS = {
    "Country Identification Rate": Metric("passed", "Identification Rate", "Identification Rate (0.0 - 1.0)"),
    "Average Score by Country": Metric("points", "Average Score", "Average Score"),
}


def score_distributions(frame: pd.DataFrame) -> None:
    st.header("Compare Model Score Distributions")
    models = st.multiselect("Select Models to Compare Score Distributions", sorted(frame["model"].unique()))
    if models:
        fig, ax = plt.subplots()
        sns.histplot(data=frame[frame["model"].isin(models)], x="points", hue="model", ax=ax, kde=True, element="step", fill=True, alpha=0.7)
        ax.set(title="Score Distributions for Selected Models", xlabel="Score", ylabel="Density")
        st.pyplot(fig)


def country_comparison(frame: pd.DataFrame) -> None:
    st.header("Country Performance Comparison")
    models, countries = sorted(frame["model"].unique()), sorted(frame["country"].unique())
    chosen_models = st.multiselect("Select Models for Country Comparison", models, default=models[:3])
    chosen_countries = st.multiselect("Select Countries for Comparison (or type to search)", countries, default=countries[:5])
    chosen = st.radio("Select Metric for Country Comparison", list(METRICS))
    metric = METRICS[chosen]
    if not (chosen_models and chosen_countries):
        st.info("Please select at least one model and one country for comparison.")
        return
    table = metric.by_country(frame[frame["model"].isin(chosen_models) & frame["country"].isin(chosen_countries)])
    fig, ax = plt.subplots(figsize=(max(10, len(chosen_countries) * 1.5), 6))
    sns.barplot(data=table, x="country", y=metric.name, hue="model", ax=ax)
    ax.set(title=f"{chosen} for Selected Models & Countries", xlabel="Country", ylabel=metric.label)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Data for Country Comparison Plot")
    st.dataframe(table)


def drill_down(frame: pd.DataFrame) -> None:
    st.header("Drill Down: Single Experiment Details")
    model = st.selectbox("Select Model for Drilldown", sorted(frame["model"].unique()))
    run = st.selectbox("Select Specific Experiment Run", sorted(frame.loc[frame["model"] == model, "run"].unique()))
    single = frame[frame["run"] == run]
    st.subheader(f"Score Distribution for Run: {run}")
    fig, ax = plt.subplots()
    single["points"].plot(kind="hist", ax=ax, bins=20, title="Score Distribution")
    ax.set(xlabel="Score", ylabel="Frequency")
    st.pyplot(fig)
    st.subheader(f"Scores per Round for Run: {run}")
    st.line_chart(single.set_index(pd.to_numeric(single["location"])).sort_index()["points"])
    st.subheader(f"Detailed Data Table for Run: {run}")
    st.dataframe(single)


def main() -> None:
    st.set_page_config(layout="wide")
    st.title("Model Results Dashboard")
    st.sidebar.header("Options")
    frame = pd.DataFrame(asdict(answer) for answer in answers(RESPONSES))
    if frame.empty:
        st.warning(f"No runs found in {RESPONSES}.")
        return
    test = st.sidebar.selectbox("Filter by Test", [ALL_TESTS, *sorted(frame["map"].unique())])
    shown = frame if test == ALL_TESTS else frame[frame["map"] == test]
    score_distributions(shown)
    country_comparison(shown)
    drill_down(shown)


if __name__ == "__main__":
    main()
