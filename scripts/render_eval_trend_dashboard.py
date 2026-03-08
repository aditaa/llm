#!/usr/bin/env python3
"""Render a lightweight HTML dashboard from supervisor eval trend TSV."""

from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class EvalTrendRow:
    run_tag: str
    step: int
    eval_rc: int
    pass_rate: float
    check_pass_rate: float
    avg_case_score: float
    regression_pass: str
    promotion_pass: str
    report_json: str


def _as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def parse_eval_trend(path: Path) -> list[EvalTrendRow]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[EvalTrendRow] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            rows.append(
                EvalTrendRow(
                    run_tag=str(raw.get("run_tag", "")),
                    step=_as_int(str(raw.get("step", ""))),
                    eval_rc=_as_int(str(raw.get("eval_rc", ""))),
                    pass_rate=_as_float(str(raw.get("pass_rate", "nan"))),
                    check_pass_rate=_as_float(str(raw.get("check_pass_rate", "nan"))),
                    avg_case_score=_as_float(str(raw.get("avg_case_score", "nan"))),
                    regression_pass=str(raw.get("regression_pass", "NA")),
                    promotion_pass=str(raw.get("promotion_pass", "NA")),
                    report_json=str(raw.get("report_json", "")),
                )
            )
    rows.sort(key=lambda row: row.step)
    return rows


def _safe_mean(values: list[float]) -> float:
    valid = [value for value in values if value == value]
    if not valid:
        return float("nan")
    return mean(valid)


def _fmt(value: float) -> str:
    if value != value:
        return "NA"
    return f"{value:.4f}"


def summarize(rows: list[EvalTrendRow]) -> dict[str, Any]:
    latest = rows[-1] if rows else None
    window = rows[-5:] if rows else []
    summary = {
        "count": len(rows),
        "latest_step": latest.step if latest else None,
        "latest_pass_rate": latest.pass_rate if latest else None,
        "latest_check_pass_rate": latest.check_pass_rate if latest else None,
        "latest_avg_case_score": latest.avg_case_score if latest else None,
        "latest_regression_pass": latest.regression_pass if latest else None,
        "latest_promotion_pass": latest.promotion_pass if latest else None,
        "rolling5_pass_rate": _safe_mean([row.pass_rate for row in window]),
        "rolling5_avg_case_score": _safe_mean([row.avg_case_score for row in window]),
    }
    return summary


def render_html(rows: list[EvalTrendRow], summary: dict[str, Any]) -> str:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{row.step}</td>"
        f"<td>{_fmt(row.pass_rate)}</td>"
        f"<td>{_fmt(row.check_pass_rate)}</td>"
        f"<td>{_fmt(row.avg_case_score)}</td>"
        f"<td>{html.escape(row.regression_pass)}</td>"
        f"<td>{html.escape(row.promotion_pass)}</td>"
        f"<td>{html.escape(Path(row.report_json).name)}</td>"
        "</tr>"
        for row in rows[-200:]
    )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Eval Trend Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; }}
    h1 {{ margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px 12px; background: #fafafa; }}
    .k {{ font-size: 12px; color: #555; }}
    .v {{ font-size: 20px; font-weight: 600; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e3e3e3; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h1>Eval Trend Dashboard</h1>
  <div class=\"grid\">
    <div class=\"card\"><div class=\"k\">Latest Step</div><div class=\"v\">{summary.get('latest_step')}</div></div>
    <div class=\"card\"><div class=\"k\">Latest Pass Rate</div><div class=\"v\">{_fmt(float(summary.get('latest_pass_rate') or float('nan')))}</div></div>
    <div class=\"card\"><div class=\"k\">Latest Avg Case Score</div><div class=\"v\">{_fmt(float(summary.get('latest_avg_case_score') or float('nan')))}</div></div>
    <div class=\"card\"><div class=\"k\">Rolling5 Pass Rate</div><div class=\"v\">{_fmt(float(summary.get('rolling5_pass_rate') or float('nan')))}</div></div>
    <div class=\"card\"><div class=\"k\">Rolling5 Avg Case Score</div><div class=\"v\">{_fmt(float(summary.get('rolling5_avg_case_score') or float('nan')))}</div></div>
    <div class=\"card\"><div class=\"k\">Promotion (Latest)</div><div class=\"v\">{html.escape(str(summary.get('latest_promotion_pass')))}</div></div>
  </div>

  <table>
    <thead>
      <tr>
        <th>Step</th>
        <th>Pass Rate</th>
        <th>Check Pass Rate</th>
        <th>Avg Case Score</th>
        <th>Regression</th>
        <th>Promotion</th>
        <th>Report</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-tsv",
        default="artifacts/reports/train_supervisor_350bt/eval_trend.tsv",
        help="Input eval trend TSV",
    )
    parser.add_argument(
        "--output-html",
        default="artifacts/reports/train_supervisor_350bt/eval_dashboard.html",
        help="Output dashboard HTML path",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/reports/train_supervisor_350bt/eval_dashboard_summary.json",
        help="Output summary JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = parse_eval_trend(Path(args.input_tsv))
    summary = summarize(rows)

    output_html = Path(args.output_html)
    output_json = Path(args.output_json)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    output_html.write_text(render_html(rows, summary), encoding="utf-8")
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"input_rows={len(rows)}")
    print(f"output_html={output_html}")
    print(f"output_json={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
