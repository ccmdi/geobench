import argparse
import json
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property
from math import comb
from pathlib import Path
from typing import Any, Self

from shungite.item import Item

MILLION = 1_000_000


def average(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def sum_recorded[T](values: Iterable[T | None], start: T) -> T | None:
    recorded = [value for value in values if value is not None]
    return sum(recorded, start) if recorded else None


class Median(StrEnum):
    MIDDLE = "middle"
    HIGH = "high"

    def __call__(self, values: Sequence[float]) -> float:
        return statistics.median_high(values) if self is Median.HIGH else statistics.median(values)


@dataclass(frozen=True)
class Measure:
    mean: float
    median: float


@dataclass(frozen=True)
class Tokens:
    input: float = 0
    output: float = 0

    @classmethod
    def of(cls, usage: dict[str, Any]) -> Self:
        return cls(usage["prompt_tokens"] + (usage.get("cached_input_tokens") or 0), usage["completion_tokens"])

    @classmethod
    def total(cls, usages: Iterable[dict[str, Any]]) -> Self:
        return sum((cls.of(usage) for usage in usages), cls())

    def __add__(self, other: Self) -> Self:
        return type(self)(self.input + other.input, self.output + other.output)

    def __truediv__(self, n: float) -> Self:
        return type(self)(self.input / n, self.output / n)

    def __bool__(self) -> bool:
        return bool(self.input or self.output)

    def __str__(self) -> str:
        return f"{self.input:,.0f} in / {self.output:,.0f} out tokens"


@dataclass(frozen=True)
class Price:
    input: float
    output: float

    def __call__(self, tokens: Tokens) -> float:
        return (tokens.input * self.input + tokens.output * self.output) / MILLION


@dataclass(frozen=True)
class Prices:
    model: Price | None = None
    judge: Price | None = None


@dataclass(frozen=True)
class Trial:
    data: dict[str, Any]
    items: dict[str, Item] | None
    model: Tokens | None
    judge: Tokens | None
    start: float | None
    end: float | None
    reply: str | None
    reasoning: str | None

    @classmethod
    def parse(cls, trace: dict[str, Any]) -> Self:
        timing = trace.get("timing", {})
        items = trace["info"].get("items")
        ends = [span["end"] for span in timing.values() if isinstance(span, dict) and span.get("end")]
        said = [node["message"] for node in trace.get("nodes", []) if node["message"]["role"] == "assistant"]
        last = said[-1] if said else {}
        return cls(
            data=trace["task"]["data"],
            items=None if items is None else {key: Item.model_validate(item) for key, item in items.items()},
            model=Tokens.of(trace["usage"]) if "usage" in trace else None,
            judge=Tokens.total(trace["extra_usage"]) if "extra_usage" in trace else None,
            start=timing.get("start"),
            end=max(ends, default=None),
            reply=last.get("content"),
            reasoning=last.get("reasoning_content"),
        )

    @property
    def task(self) -> str:
        return self.data["id"]

    @property
    def seconds(self) -> float | None:
        return None if self.start is None or self.end is None else self.end - self.start


@dataclass(frozen=True)
class Setup:
    benchmark: str
    model: str
    sampling: str
    median: Median = Median.MIDDLE

    @classmethod
    def of(cls, config: dict[str, Any]) -> Self:
        sampling = None if config.get("sampling") is None else {key: value for key, value in config["sampling"].items() if value is not None}
        median = Median(config.get("report", {}).get("median", Median.MIDDLE))
        return cls(config["env"]["taskset"]["id"], config["model"], json.dumps(sampling, sort_keys=True), median)

    @property
    def settings(self) -> dict[str, Any] | None:
        return json.loads(self.sampling)

    def __str__(self) -> str:
        settings = ", ".join(
            f"{key} {value if isinstance(value, str) else json.dumps(value)}" for key, value in (self.settings or {}).items()
        )
        return f"{self.model} ({settings})" if settings else self.model


@dataclass(frozen=True)
class Run:
    setup: Setup
    trials: tuple[Trial, ...]

    @classmethod
    def of(cls, config: dict[str, Any], episodes: Iterable[dict[str, Any]]) -> Self:
        return cls(Setup.of(config), tuple(Trial.parse(trace) for episode in episodes for trace in episode["traces"]))

    @classmethod
    def read(cls, dir: Path) -> Self:
        config = json.loads((dir / "configs" / "resolved" / "run.json").read_text())
        return cls.of(config, (json.loads(line) for line in (dir / "traces.jsonl").read_text("utf-8").split("\n") if line.strip()))

    @property
    def elapsed(self) -> float | None:
        timed = [t for t in self.trials if t.seconds is not None]
        return max(t.end for t in timed) - min(t.start for t in timed) if timed else None


@dataclass(frozen=True)
class Outcomes:
    task: str
    key: str
    trials: tuple[Item, ...]

    @property
    def kind(self) -> str | None:
        return self.trials[0].kind

    @property
    def decided(self) -> list[Item]:
        return [trial for trial in self.trials if trial.decided]

    @property
    def passes(self) -> int:
        return sum(bool(trial.passed) for trial in self.trials)

    @property
    def unknown(self) -> int:
        return len(self.trials) - len(self.decided)

    @property
    def pass_rate(self) -> float:
        return self.passes / len(self.decided)

    @property
    def mean_score(self) -> float:
        return statistics.fmean(trial.score for trial in self.decided if trial.score is not None)

    def pass_at(self, k: int) -> float:
        n = len(self.decided)
        return 1 - comb(n - self.passes, k) / comb(n, k)

    def pass_all(self, k: int) -> float:
        return comb(self.passes, k) / comb(len(self.decided), k)


@dataclass(frozen=True)
class Result:
    setup: Setup
    runs: tuple[Run, ...]

    @cached_property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(trial for run in self.runs for trial in run.trials)

    @cached_property
    def items(self) -> tuple[Outcomes, ...]:
        grouped: dict[tuple[str, str], list[Item]] = {}
        for trial in self.trials:
            for key, item in (trial.items or {}).items():
                grouped.setdefault((trial.task, key), []).append(item)
        if not grouped:
            raise ValueError(f"{self.setup}: no trial was scored")
        return tuple(Outcomes(task, key, tuple(items)) for (task, key), items in grouped.items())

    @property
    def trials_per_task(self) -> tuple[int, int]:
        counts = Counter(trial.task for trial in self.trials).values()
        return min(counts), max(counts)

    @property
    def k(self) -> int:
        return min(len(outcomes.trials) for outcomes in self.items)

    @property
    def decided(self) -> list[Outcomes]:
        return [outcomes for outcomes in self.items if outcomes.decided]

    @property
    def comparable(self) -> list[Outcomes]:
        return [outcomes for outcomes in self.items if len(outcomes.decided) >= self.k]

    @property
    def pass_rate(self) -> float | None:
        return average(outcomes.pass_rate for outcomes in self.decided)

    @property
    def pass_at_k(self) -> float | None:
        return average(outcomes.pass_at(self.k) for outcomes in self.comparable)

    @property
    def pass_hat_k(self) -> float | None:
        return average(outcomes.pass_all(self.k) for outcomes in self.comparable)

    @property
    def mean_score(self) -> float | None:
        return average(outcomes.mean_score for outcomes in self.decided)

    @property
    def trial_means(self) -> list[float]:
        rounds = max(len(outcomes.trials) for outcomes in self.items)
        means = (
            average(o.trials[r].score for o in self.items if r < len(o.trials) and o.trials[r].score is not None)
            for r in range(rounds)
        )
        return [mean for mean in means if mean is not None]

    @property
    def measures(self) -> dict[str, Measure]:
        values: dict[str, list[float]] = {}
        for outcomes in self.items:
            for item in outcomes.trials:
                for name, value in (item.model_extra or {}).items():
                    if isinstance(value, int | float):
                        values.setdefault(name, []).append(float(value))
        return {name: Measure(statistics.fmean(v), self.setup.median(v)) for name, v in sorted(values.items())}

    @property
    def unknown(self) -> int:
        return sum(outcomes.unknown for outcomes in self.items)

    @property
    def errored(self) -> int:
        return sum(trial.items is None for trial in self.trials)

    @property
    def model_tokens(self) -> Tokens | None:
        return sum_recorded((trial.model for trial in self.trials), Tokens())

    @property
    def judge_tokens(self) -> Tokens | None:
        return sum_recorded((trial.judge for trial in self.trials), Tokens())

    @property
    def full_runs(self) -> float:
        return len(self.trials) / len(Counter(trial.task for trial in self.trials))

    @property
    def elapsed(self) -> float | None:
        return sum_recorded((run.elapsed for run in self.runs), 0.0)

    @property
    def seconds_per_trial(self) -> float | None:
        return average(trial.seconds for trial in self.trials if trial.seconds is not None)

    @property
    def run_dates(self) -> list[str]:
        starts = [[trial.start for trial in run.trials if trial.start is not None] for run in self.runs]
        return sorted({datetime.fromtimestamp(min(s), UTC).date().isoformat() for s in starts if s})

    def bill(self, prices: Prices) -> dict[str, tuple[Tokens | None, float | None]]:
        return {
            who: (spent, None if spent is None or price is None else price(spent))
            for who, spent, price in (
                ("model", self.model_tokens and self.model_tokens / self.full_runs, prices.model),
                ("judge", self.judge_tokens and self.judge_tokens / self.full_runs, prices.judge),
            )
        }

    def cost_cell(self, prices: Prices) -> str:
        return " + ".join(
            ("" if who == "model" else f"{who} ") + ("-" if spent is None else str(spent) if cost is None else f"${cost:,.2f}")
            for who, (spent, cost) in self.bill(prices).items()
            if spent or who == "model"
        )

    def as_json(self, prices: Prices) -> dict[str, Any]:
        return {
            "model": self.setup.model,
            "sampling": self.setup.settings,
            "run_dates": self.run_dates,
            "trials_per_task": list(self.trials_per_task),
            "k": self.k,
            "pass_rate": self.pass_rate,
            "pass_at_k": self.pass_at_k,
            "pass_hat_k": self.pass_hat_k,
            "mean_score": self.mean_score,
            "trial_means": self.trial_means,
            "measures": {name: vars(measure) for name, measure in self.measures.items()},
            "unknown": self.unknown,
            "errored": self.errored,
            "usage": {
                "model": None if self.model_tokens is None else vars(self.model_tokens),
                "judge": None if self.judge_tokens is None else vars(self.judge_tokens),
                "trials": len(self.trials),
                "full_runs": self.full_runs,
                "elapsed_seconds": self.elapsed,
                "seconds_per_trial": self.seconds_per_trial,
            },
            "cost_per_full_run": {who: cost for who, (_, cost) in self.bill(prices).items()},
            "items": [
                {
                    "task": o.task,
                    "item": o.key,
                    "kind": o.kind,
                    "scores": [trial.score for trial in o.trials],
                    "passed": [trial.passed for trial in o.trials],
                }
                for o in self.items
            ],
        }


def shown(value: float | None, template: str) -> str:
    return "-" if value is None else template.format(value)


def span(low: float, high: float, format: str = "") -> str:
    return f"{low:{format}}" if low == high else f"{low:{format}}-{high:{format}}"


def table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    return [" | ".join(header), " | ".join("---" for _ in header), *(" | ".join(map(str, row)) for row in rows)]


@dataclass(frozen=True)
class Report:
    benchmark: str
    results: tuple[Result, ...]

    @classmethod
    def of(cls, runs: Iterable[Run]) -> Self:
        grouped: dict[Setup, list[Run]] = {}
        for run in runs:
            grouped.setdefault(run.setup, []).append(run)
        benchmarks = {setup.benchmark for setup in grouped}
        if len(benchmarks) != 1:
            raise ValueError(f"a report covers one benchmark, got {sorted(benchmarks)}")
        return cls(benchmarks.pop(), tuple(Result(setup, tuple(runs)) for setup, runs in grouped.items()))

    @property
    def models(self) -> set[str]:
        return {result.setup.model for result in self.results}

    def as_json(self, prices: Prices) -> dict[str, Any]:
        return {"benchmark": self.benchmark, "results": [result.as_json(prices) for result in self.results]}

    def markdown(self, prices: Prices) -> str:
        extra = [
            (name, cell)
            for name, cell, shown in (
                ("Unknown", lambda r: r.unknown, any(r.unknown for r in self.results)),
                ("Errored", lambda r: r.errored, any(r.errored for r in self.results)),
            )
            if shown
        ]
        summary = table(
            [
                "Model (version, settings)",
                "Trials per task",
                "Pass rate",
                "pass^k",
                "Cost per full run",
                "Mean score (min-max over trials)",
                "pass@k",
                "k",
                "Run date",
                *(name for name, _ in extra),
            ],
            (
                [
                    r.setup,
                    span(*r.trials_per_task),
                    shown(r.pass_rate, "{:.1%}"),
                    shown(r.pass_hat_k, "{:.1%}"),
                    r.cost_cell(prices),
                    "-" if r.mean_score is None else f"{r.mean_score:.3f} ({span(min(r.trial_means), max(r.trial_means), '.3f')})",
                    shown(r.pass_at_k, "{:.1%}"),
                    r.k,
                    ", ".join(r.run_dates),
                    *(cell(r) for _, cell in extra),
                ]
                for r in self.results
            ),
        )
        sections = [f"# {self.benchmark}", "", *summary]
        for r in self.results:
            usage = [("Model", r.model_tokens), *([("Judge", r.judge_tokens)] if r.judge_tokens else [])]
            trials = len(r.trials)
            sections += [
                "",
                f"## {r.setup}",
                "",
                *table(
                    ["Usage", "Total", "Per trial"],
                    [
                        *(
                            (f"{who} {side} tokens", shown(spent, "{:,.0f}"), shown(None if spent is None else spent / trials, "{:,.1f}"))
                            for who, tokens in usage
                            for side in ("input", "output")
                            for spent in [None if tokens is None else getattr(tokens, side)]
                        ),
                        ("Wall time", shown(r.elapsed, "{:,.1f} s"), shown(r.seconds_per_trial, "{:,.1f} s")),
                    ],
                ),
                *(
                    ["", *table(["Measure", "Mean", "Median"], ((name, f"{m.mean:.4g}", f"{m.median:.4g}") for name, m in r.measures.items()))]
                    if r.measures
                    else []
                ),
                "",
                *table(
                    ["Task", "Item", "Kind", "Scores", "Passed"],
                    (
                        [
                            o.task,
                            o.key,
                            o.kind or "-",
                            ", ".join("?" if t.score is None else f"{t.score:.4g}" for t in o.trials),
                            f"{o.passes}/{len(o.decided)}",
                        ]
                        for o in r.items
                    ),
                ),
            ]
        sections += [
            "",
            "pass@k: share of items that pass at least once in k trials. pass^k: share that pass in all k. "
            "k is the fewest trials any item got; an item with more uses the unbiased estimator (exact when n = k), "
            "and one left with fewer decided trials by UNKNOWN verdicts (`?`) counts in neither. "
            "Pass rate and mean score average over items; the spread is the min-max of the per-trial mean score.",
        ]
        return "\n".join(sections)


def price(parser: argparse.ArgumentParser, flag: str, input: float | None, output: float | None) -> Price | None:
    if input is None and output is None:
        return None
    if input is None or output is None:
        parser.error(f"{flag}-in and {flag}-out go together")
    return Price(input, output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m shungite.report",
        description="Pass rate, pass@k, pass^k, score, tokens and cost per model from saved vf-run traces.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="vf-run output directories")
    for flag, whose in (("--price", "model"), ("--judge-price", "judge")):
        parser.add_argument(f"{flag}-in", type=float, help=f"{whose} input price, USD per 1M tokens")
        parser.add_argument(f"{flag}-out", type=float, help=f"{whose} output price, USD per 1M tokens")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)
    prices = Prices(
        price(parser, "--price", args.price_in, args.price_out),
        price(parser, "--judge-price", args.judge_price_in, args.judge_price_out),
    )
    report = Report.of(Run.read(run) for run in args.runs)
    if prices.model is not None and len(report.models) > 1:
        parser.error(f"--price-in/--price-out price one model, the runs hold {sorted(report.models)}")
    print(json.dumps(report.as_json(prices), indent=2) if args.json else report.markdown(prices))


if __name__ == "__main__":
    main()
