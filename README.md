GeoBench is a benchmark for evaluating how well large language models can geolocate images, through the context of GeoGuessr. This project tests whether models can generalize beyond their primary training modalities to perform spatial reasoning tasks.

# **[Leaderboard](https://geobench.org)**

![](img/leaderboard.png)
For an in-depth explanation of the results, covering things like model behavior and reasoning, see my [writeup](https://ccmdi.com/blog/GeoBench).

# Installation
Needs [uv](https://docs.astral.sh/uv/).
```bash
git clone https://github.com/ccmdi/geobench.git
cd geobench
uv sync
cp .env.example .env
```
Fill in the key for each provider you want to test in `.env`.

## Test a model
```bash
uv run geobench <model> --map <map>
```
Add `-n <locations>` to run a random few.

Each location scores 0 to 5000 points on the GeoGuessr curve, scaled to the map's size. The reward is points / 5000, and `country_correct` tracks whether the country was right.

## Results
Build the report from one or more run folders. It only reads the saved traces, so no model gets called:
```bash
uv run python -m shungite.report responses/<run> [responses/<run> ...]
```
You get Markdown tables: per model, the pass rate, pass^k, pass@k, mean score (points / 5000) with its spread across trials, tokens and wall time, then every location's score in each trial. `--price-in` / `--price-out` (USD per 1M tokens) turn the tokens into a cost, and `--json` prints the numbers as JSON. A location passes when the country is right.

## Create a dataset
```bash
uv run dataset --num <n> --output <map> --map <geoguessr map id>
```
This needs `NCFA` in `.env`. Instructions for setting it up can be found [here](https://github.com/EvickaStudio/GeoGuessr-API?tab=readme-ov-file#authentication).

## Compare guesses
Run `uv run viz` to browse and compare runs' guesses in your browser.
