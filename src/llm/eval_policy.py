"""Evaluation report comparison and checkpoint promotion policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalSummary:
    pass_rate: float
    check_pass_rate: float
    avg_case_score: float
    cases_total: int
    cases_passed: int
    checks_total: int
    checks_passed: int


def _to_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {field}: {value!r}") from exc


def _to_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid int for {field}: {value!r}") from exc


def summary_from_report(report: dict[str, Any]) -> EvalSummary:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("report is missing summary object")
    return EvalSummary(
        pass_rate=_to_float(summary.get("pass_rate"), "pass_rate"),
        check_pass_rate=_to_float(summary.get("check_pass_rate"), "check_pass_rate"),
        avg_case_score=_to_float(summary.get("avg_case_score"), "avg_case_score"),
        cases_total=_to_int(summary.get("cases_total"), "cases_total"),
        cases_passed=_to_int(summary.get("cases_passed"), "cases_passed"),
        checks_total=_to_int(summary.get("checks_total"), "checks_total"),
        checks_passed=_to_int(summary.get("checks_passed"), "checks_passed"),
    )


def compare_summaries(current: EvalSummary, baseline: EvalSummary) -> dict[str, float]:
    return {
        "pass_rate_delta": current.pass_rate - baseline.pass_rate,
        "check_pass_rate_delta": current.check_pass_rate - baseline.check_pass_rate,
        "avg_case_score_delta": current.avg_case_score - baseline.avg_case_score,
    }


def evaluate_promotion_policy(
    *,
    current: EvalSummary,
    baseline: EvalSummary | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    absolute = policy.get("absolute", {})
    regression = policy.get("regression", {})
    improvement = policy.get("improvement", {})
    if not isinstance(absolute, dict):
        raise ValueError("promotion policy 'absolute' must be an object")
    if not isinstance(regression, dict):
        raise ValueError("promotion policy 'regression' must be an object")
    if not isinstance(improvement, dict):
        raise ValueError("promotion policy 'improvement' must be an object")

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def add_check(name: str, enabled: bool, passed: bool | None, detail: dict[str, Any]) -> None:
        row = {"name": name, "enabled": enabled, "pass": passed}
        row.update(detail)
        checks.append(row)
        if enabled and passed is False:
            failures.append(name)

    for field, metric in (
        ("min_pass_rate", current.pass_rate),
        ("min_check_pass_rate", current.check_pass_rate),
        ("min_avg_case_score", current.avg_case_score),
    ):
        if field in absolute:
            threshold = _to_float(absolute.get(field), field)
            add_check(
                field,
                True,
                metric >= threshold,
                {"expected_min": threshold, "observed": metric},
            )

    if baseline is None:
        for field in ("max_pass_rate_drop", "max_check_pass_rate_drop", "max_avg_case_score_drop"):
            if field in regression:
                add_check(
                    field,
                    True,
                    False,
                    {
                        "error": "baseline_report_required",
                        "expected_max_drop": _to_float(regression.get(field), field),
                    },
                )
        for field in ("min_pass_rate_gain", "min_check_pass_rate_gain", "min_avg_case_score_gain"):
            if field in improvement:
                add_check(
                    field,
                    True,
                    False,
                    {
                        "error": "baseline_report_required",
                        "expected_min_gain": _to_float(improvement.get(field), field),
                    },
                )
        deltas: dict[str, float] | None = None
    else:
        deltas = compare_summaries(current, baseline)
        for field, delta_key in (
            ("max_pass_rate_drop", "pass_rate_delta"),
            ("max_check_pass_rate_drop", "check_pass_rate_delta"),
            ("max_avg_case_score_drop", "avg_case_score_delta"),
        ):
            if field in regression:
                max_drop = _to_float(regression.get(field), field)
                add_check(
                    field,
                    True,
                    deltas[delta_key] >= -max_drop,
                    {"expected_max_drop": max_drop, "observed_delta": deltas[delta_key]},
                )

        for field, delta_key in (
            ("min_pass_rate_gain", "pass_rate_delta"),
            ("min_check_pass_rate_gain", "check_pass_rate_delta"),
            ("min_avg_case_score_gain", "avg_case_score_delta"),
        ):
            if field in improvement:
                min_gain = _to_float(improvement.get(field), field)
                add_check(
                    field,
                    True,
                    deltas[delta_key] >= min_gain,
                    {"expected_min_gain": min_gain, "observed_delta": deltas[delta_key]},
                )

    enabled_count = sum(1 for row in checks if bool(row["enabled"]))
    promoted = enabled_count > 0 and not failures
    return {
        "policy_name": str(policy.get("name", "promotion_policy")),
        "promoted": promoted,
        "enabled_checks": enabled_count,
        "failed_checks": failures,
        "checks": checks,
        "deltas": deltas,
    }

