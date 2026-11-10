import json
from pathlib import Path
from typing import Any

import pytest

from shungite import Item
from shungite.report import Measure, Price, Prices, Report, Run, main

DAY = 1_794_052_800.0
P, F, U = (1.0, True), (0.0, False), (None, None)


def trace(task: str, items: dict[str, tuple[float | None, bool | None]] | None, start: float = DAY, judge: int = 0) -> dict[str, Any]:
    info = {} if items is None else {"items": {key: {"score": s, "passed": p, "kind": "location"} for key, (s, p) in items.items()}}
    return {
        "task": {"type": "Task", "data": {"id": task}},
        "info": info,
        "usage": {"prompt_tokens": 80, "cached_input_tokens": 20, "completion_tokens": 50},
        "extra_usage": [{"prompt_tokens": 10, "completion_tokens": 2}] * judge,
        "timing": {"start": start, "agent": {"start": start, "end": start + 3.0, "duration": 3.0}, "scoring": {"start": start + 3.0, "end": start + 4.0}},
    }


def write_run(
    dir: Path, traces: list[dict[str, Any]], model: str = "lab/model-1", benchmark: str = "bench", report: dict[str, str] | None = None, **sampling: object
) -> Path:
    (dir / "configs" / "resolved").mkdir(parents=True)
    config = {"env": {"taskset": {"id": benchmark}}, "model": model, "sampling": {"temperature": None, "max_tokens": None} | sampling}
    config |= {} if report is None else {"report": report}
    (dir / "configs" / "resolved" / "run.json").write_text(json.dumps(config))
    (dir / "traces.jsonl").write_text("".join(json.dumps({"traces": [t]}) + "\n" for t in traces))
    return dir


def report(*dirs: Path) -> Report:
    return Report.of(Run.read(dir) for dir in dirs)


def three_trials(tmp_path: Path) -> Path:
    outcomes = {"a": [P, P, P], "b": [P, F, F], "c": [F, F, F]}
    traces = [trace(task, {"1": outcomes[task][r]}, start=DAY + r) for task in outcomes for r in range(3)]
    return write_run(tmp_path / "k3", traces, temperature=0.4, reasoning_effort="high")


def test_one_trial_makes_pass_at_k_and_pass_hat_k_the_pass_rate(tmp_path: Path) -> None:
    [result] = report(write_run(tmp_path, [trace("a", {"1": P, "2": F}), trace("b", {"1": (0.8, True)})])).results
    assert result.k == 1
    assert result.pass_rate == pytest.approx(2 / 3)
    assert result.pass_at_k == pytest.approx(2 / 3)
    assert result.pass_hat_k == pytest.approx(2 / 3)
    assert result.mean_score == pytest.approx(1.8 / 3)
    assert result.trial_means == [pytest.approx(1.8 / 3)]


def test_three_trials(tmp_path: Path) -> None:
    [result] = report(three_trials(tmp_path)).results
    assert result.trials_per_task == (3, 3)
    assert result.k == 3
    assert result.pass_at_k == pytest.approx(2 / 3)
    assert result.pass_hat_k == pytest.approx(1 / 3)
    assert result.pass_rate == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert result.mean_score == pytest.approx(4 / 9)
    assert result.trial_means == [pytest.approx(2 / 3), pytest.approx(1 / 3), pytest.approx(1 / 3)]
    assert [(o.task, o.passes, len(o.decided)) for o in result.items] == [("a", 3, 3), ("b", 1, 3), ("c", 0, 3)]


def test_unknown_counts_as_neither_pass_nor_fail(tmp_path: Path) -> None:
    traces = [trace("a", {"1": o}) for o in (P, U, F)] + [trace("b", {"1": o}) for o in (P, P, P)]
    [result] = report(write_run(tmp_path, traces)).results
    assert result.unknown == 1
    assert result.k == 3
    assert [o.task for o in result.comparable] == ["b"]
    assert result.pass_at_k == 1.0
    assert result.pass_hat_k == 1.0
    assert result.pass_rate == pytest.approx((1 / 2 + 1) / 2)
    assert result.mean_score == pytest.approx((0.5 + 1) / 2)


def test_an_item_with_only_unknown_trials_is_left_out(tmp_path: Path) -> None:
    [result] = report(write_run(tmp_path, [trace("a", {"1": U}), trace("b", {"1": P})])).results
    assert result.unknown == 1
    assert result.pass_rate == 1.0
    assert result.pass_at_k == 1.0


def test_varying_trial_counts_use_the_unbiased_estimator(tmp_path: Path) -> None:
    first = write_run(tmp_path / "first", [trace("x", {"1": P}), trace("x", {"1": F}), trace("y", {"1": P}), trace("y", {"1": P})])
    second = write_run(tmp_path / "second", [trace("x", {"1": F})])
    [result] = report(first, second).results
    assert result.trials_per_task == (2, 3)
    assert result.k == 2
    assert result.pass_at_k == pytest.approx((1 - 1 / 3 + 1) / 2)
    assert result.pass_hat_k == pytest.approx((0 + 1) / 2)


def test_runs_group_by_model_and_settings(tmp_path: Path) -> None:
    hot = write_run(tmp_path / "hot", [trace("a", {"1": P})], temperature=1.0)
    cold = write_run(tmp_path / "cold", [trace("a", {"1": F})], temperature=0.0)
    other = write_run(tmp_path / "other", [trace("a", {"1": P})], model="lab/model-2", temperature=1.0)
    assert [(str(r.setup), r.pass_rate) for r in report(hot, cold, other).results] == [
        ("lab/model-1 (temperature 1.0)", 1.0),
        ("lab/model-1 (temperature 0.0)", 0.0),
        ("lab/model-2 (temperature 1.0)", 1.0),
    ]


def test_one_report_holds_one_benchmark(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", [trace("a", {"1": P})])
    b = write_run(tmp_path / "b", [trace("a", {"1": P})], benchmark="other")
    with pytest.raises(ValueError, match="one benchmark"):
        report(a, b)


def test_usage_and_cost_per_full_run(tmp_path: Path) -> None:
    [result] = report(write_run(tmp_path, [trace("a", {"1": P}, judge=2), trace("a", {"1": P}, judge=2), trace("b", {"1": F})])).results
    assert result.full_runs == 1.5
    assert vars(result.model_tokens) == {"input": 300, "output": 150}
    assert vars(result.judge_tokens) == {"input": 40, "output": 8}
    assert result.cost_cell(Prices()) == "200 in / 100 out tokens + judge 27 in / 5 out tokens"
    priced = Prices(Price(input=3.0, output=15.0), Price(input=1.0, output=5.0))
    assert {who: cost for who, (_, cost) in result.bill(priced).items()} == {"model": pytest.approx((200 * 3 + 100 * 15) / 1e6), "judge": pytest.approx((40 + 8 * 5) / 1.5 / 1e6)}
    assert result.cost_cell(Prices(Price(input=3000.0, output=15000.0))) == "$2.10 + judge 27 in / 5 out tokens"
    assert result.seconds_per_trial == 4.0
    assert result.elapsed == 4.0
    assert result.run_dates == ["2026-11-07"]


def test_a_trial_that_never_scored_counts_as_errored(tmp_path: Path) -> None:
    [result] = report(write_run(tmp_path, [trace("a", {"1": P}), trace("a", None)])).results
    assert result.errored == 1
    assert result.trials_per_task == (2, 2)
    assert result.k == 1


def test_markdown(tmp_path: Path) -> None:
    text = report(three_trials(tmp_path)).markdown(Prices())
    lines = text.splitlines()
    assert lines[0] == "# bench"
    assert lines[2] == (
        "Model (version, settings) | Trials per task | Pass rate | pass^k | Cost per full run"
        " | Mean score (min-max over trials) | pass@k | k | Run date"
    )
    assert lines[4] == (
        "lab/model-1 (reasoning_effort high, temperature 0.4) | 3 | 44.4% | 33.3% | 300 in / 150 out tokens"
        " | 0.444 (0.333-0.667) | 66.7% | 3 | 2026-11-07"
    )
    assert "b | 1 | location | 1, 0, 0 | 1/3" in lines
    assert "Model input tokens | 900 | 100.0" in lines
    assert "Unknown" not in text


def test_markdown_shows_unknown_only_when_present(tmp_path: Path) -> None:
    text = report(write_run(tmp_path, [trace("a", {"1": P}), trace("a", {"1": U})])).markdown(Prices())
    assert text.splitlines()[2].endswith("| Run date | Unknown")
    assert "a | 1 | location | 1, ? | 1/1" in text.splitlines()


def test_cli_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main([str(three_trials(tmp_path)), "--json", "--price-in", "3", "--price-out", "15"])
    [result] = json.loads(capsys.readouterr().out)["results"]
    assert result["k"] == 3
    assert result["pass_hat_k"] == pytest.approx(1 / 3)
    assert result["sampling"] == {"reasoning_effort": "high", "temperature": 0.4}
    assert result["cost_per_full_run"] == {"model": pytest.approx((300 * 3 + 150 * 15) / 1e6), "judge": None}
    assert result["items"][1] == {"task": "b", "item": "1", "kind": "location", "scores": [1.0, 0.0, 0.0], "passed": [True, False, False]}


def test_cli_refuses_one_price_for_two_models(tmp_path: Path) -> None:
    a = write_run(tmp_path / "a", [trace("a", {"1": P})])
    b = write_run(tmp_path / "b", [trace("a", {"1": P})], model="lab/model-2")
    with pytest.raises(SystemExit):
        main([str(a), str(b), "--price-in", "1", "--price-out", "1"])


def test_cli_refuses_half_a_price(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(three_trials(tmp_path)), "--price-in", "1"])


@pytest.mark.parametrize(("score", "passed"), [(None, True), (1.0, None), (1.5, True), (-0.1, False)])
def test_an_item_is_decided_as_a_whole_and_scores_in_unit_range(score: float | None, passed: bool | None) -> None:
    with pytest.raises(ValueError):
        Item(score=score, passed=passed)


def test_an_item_keeps_its_extra_fields() -> None:
    item = Item(score=0.5, passed=False, kind="location", km=0.7)
    assert item.model_dump() == {"score": 0.5, "passed": False, "kind": "location", "km": 0.7}


def test_numeric_item_fields_report_their_mean_and_median(tmp_path: Path) -> None:
    traces = [trace(task, {"1": P}) for task in "abcde"]
    for t, km in zip(traces, [1.0, 2.0, 3.0, 5.0, None]):
        t["info"]["items"]["1"] |= {"refused": km is None} | ({} if km is None else {"km": km})
    [middle] = report(write_run(tmp_path / "middle", traces)).results
    [high] = report(write_run(tmp_path / "high", traces, report={"median": "high"})).results
    assert middle.measures == {"km": Measure(2.75, 2.5), "refused": Measure(0.2, 0.0)}
    assert high.measures["km"] == Measure(2.75, 3.0)
    assert high.as_json(Prices())["measures"]["km"] == {"mean": 2.75, "median": 3.0}


def test_unrecorded_usage_timing_and_sampling_read_as_blank(tmp_path: Path) -> None:
    bare = {"task": {"type": "Task", "data": {"id": "a"}}, "info": {"items": {"1": {"score": 1.0, "passed": True}}}}
    dir = write_run(tmp_path, [bare, bare | {"timing": {"start": DAY}}])
    (dir / "configs" / "resolved" / "run.json").write_text(json.dumps({"env": {"taskset": {"id": "bench"}}, "model": "lab/model-1"}))
    [result] = report(dir).results
    assert (result.model_tokens, result.judge_tokens, result.elapsed, result.seconds_per_trial) == (None, None, None, None)
    assert result.run_dates == ["2026-11-07"]
    assert result.cost_cell(Prices(Price(input=1.0, output=1.0))) == "-"
    assert str(result.setup) == "lab/model-1"
    shown = result.as_json(Prices())
    assert (shown["sampling"], shown["usage"]["model"], shown["usage"]["elapsed_seconds"]) == (None, None, None)
    text = report(dir).markdown(Prices())
    assert "Model input tokens | - | -" in text
    assert "Wall time | - | -" in text


def test_traces_split_only_on_newlines(tmp_path: Path) -> None:
    line = trace("a", {"1": P}) | {"last_reply": "a\u2028b\x85c"}
    dir = tmp_path / "run"
    write_run(dir, [line])
    (dir / "traces.jsonl").write_text(json.dumps({"traces": [line]}, ensure_ascii=False) + "\n", "utf-8")
    assert len(report(dir).results[0].trials) == 1
