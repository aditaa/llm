#!/usr/bin/env python3
"""Run a standardized prompt-suite evaluation against a checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from llm.eval_policy import compare_summaries, evaluate_promotion_policy, summary_from_report
from llm.generate import _resolve_device, _sample_next_token
from llm.model import GPTModel, model_config_from_dict
from llm.tokenizer import TokenizerLike, load_tokenizer

URL_PATTERN = re.compile(r"(https?://|www\.)", flags=re.IGNORECASE)


@dataclass
class Runtime:
    checkpoint_path: Path
    tokenizer_path: Path
    step: int
    model: GPTModel
    tokenizer: TokenizerLike
    device: torch.device
    max_seq_len: int
    eos_id: int


def _max_repeated_char_run(text: str) -> int:
    if not text:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(text)):
        if text[idx] == text[idx - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text)


def _extract_completion(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].lstrip()
    return full_text


def _load_runtime(checkpoint_path: Path, device_arg: str) -> Runtime:
    device = _resolve_device(device_arg)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cfg_raw = checkpoint.get("model_config")
    if not isinstance(model_cfg_raw, dict):
        raise ValueError("checkpoint missing model_config")
    model_cfg = model_config_from_dict(model_cfg_raw)

    tokenizer_path_raw = checkpoint.get("tokenizer_path")
    if not isinstance(tokenizer_path_raw, str):
        raise ValueError("checkpoint missing tokenizer_path")
    tokenizer_path = Path(tokenizer_path_raw)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer path missing: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return Runtime(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        step=int(checkpoint.get("step", 0)),
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_seq_len=int(model_cfg.max_seq_len),
        eos_id=int(tokenizer.eos_id if tokenizer.eos_id is not None else -1),
    )


def _generate(
    runtime: Runtime,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    stop_on_eos: bool,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    token_ids = runtime.tokenizer.encode(prompt)
    if not token_ids:
        bos_id = runtime.tokenizer.bos_id if runtime.tokenizer.bos_id is not None else 0
        token_ids = [bos_id]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = token_ids[-runtime.max_seq_len :]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=runtime.device)
            logits, _ = runtime.model(input_ids)
            next_tok = _sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            next_id = int(next_tok.item())
            token_ids.append(next_id)
            if stop_on_eos and next_id == runtime.eos_id:
                break

    full_text = runtime.tokenizer.decode(token_ids, skip_special_tokens=True)
    completion = _extract_completion(full_text, prompt)
    return {
        "prompt": prompt,
        "full_text": full_text,
        "completion": completion,
        "token_count": len(token_ids),
    }


def _run_checks(completion: str, checks: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    results: list[dict[str, Any]] = []

    min_chars = int(checks.get("min_completion_chars", 1))
    has_min_chars = len(completion) >= min_chars
    results.append(
        {
            "name": "min_completion_chars",
            "pass": has_min_chars,
            "expected": min_chars,
            "observed": len(completion),
        }
    )

    expect_any = list(checks.get("expect_any", []))
    if expect_any:
        lower = completion.lower()
        matched = [item for item in expect_any if str(item).lower() in lower]
        results.append(
            {
                "name": "expect_any",
                "pass": len(matched) > 0,
                "expected": expect_any,
                "matched": matched,
            }
        )

    reject_any = list(checks.get("reject_any", []))
    if reject_any:
        lower = completion.lower()
        hits = [item for item in reject_any if str(item).lower() in lower]
        results.append(
            {
                "name": "reject_any",
                "pass": len(hits) == 0,
                "rejected": reject_any,
                "hits": hits,
            }
        )

    if "max_url_count" in checks:
        max_url_count = int(checks["max_url_count"])
        url_count = len(URL_PATTERN.findall(completion))
        results.append(
            {
                "name": "max_url_count",
                "pass": url_count <= max_url_count,
                "expected_max": max_url_count,
                "observed": url_count,
            }
        )

    if "max_repeated_char_run" in checks:
        max_run = int(checks["max_repeated_char_run"])
        observed_run = _max_repeated_char_run(completion)
        results.append(
            {
                "name": "max_repeated_char_run",
                "pass": observed_run <= max_run,
                "expected_max": max_run,
                "observed": observed_run,
            }
        )

    if "max_non_ascii_ratio" in checks:
        max_ratio = float(checks["max_non_ascii_ratio"])
        observed_ratio = _non_ascii_ratio(completion)
        results.append(
            {
                "name": "max_non_ascii_ratio",
                "pass": observed_ratio <= max_ratio,
                "expected_max": max_ratio,
                "observed": observed_ratio,
            }
        )

    case_pass = all(bool(row["pass"]) for row in results) if results else True
    return results, case_pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path (*.pt)")
    parser.add_argument(
        "--suite",
        default="configs/eval/standard_prompt_suite_v2.json",
        help="Prompt suite JSON path",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output report JSON path (default: artifacts/reports/evals/<name>_<utc>.json)",
    )
    parser.add_argument("--device", default="auto", help="Torch device (auto, cpu, cuda, cuda:0...)")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Max new tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k cutoff (0 disables top-k)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--seed-stride", type=int, default=97, help="Per-case seed offset")
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Optional baseline eval report JSON to compare against",
    )
    parser.add_argument(
        "--promotion-policy",
        default=None,
        help="Optional promotion policy JSON path (absolute/regression/improvement checks)",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if baseline comparison shows metric regressions beyond limits",
    )
    parser.add_argument(
        "--max-pass-rate-drop",
        type=float,
        default=0.0,
        help="Allowed baseline pass_rate drop before regression failure (default: 0.0)",
    )
    parser.add_argument(
        "--max-check-pass-rate-drop",
        type=float,
        default=0.0,
        help="Allowed baseline check_pass_rate drop before regression failure (default: 0.0)",
    )
    parser.add_argument(
        "--max-avg-case-score-drop",
        type=float,
        default=0.0,
        help="Allowed baseline avg_case_score drop before regression failure (default: 0.0)",
    )
    parser.add_argument(
        "--fail-on-no-promotion",
        action="store_true",
        help="Exit with code 1 when a promotion policy is provided and promotion fails",
    )
    parser.add_argument(
        "--fail-below-pass-rate",
        type=float,
        default=None,
        help="Exit with code 1 if suite pass_rate is below this threshold",
    )
    parser.add_argument("--keep-full-text", action="store_true", help="Store full prompt+output text")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    suite_path = Path(args.suite)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not suite_path.exists():
        raise FileNotFoundError(suite_path)
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be > 0")
    if args.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if args.top_k < 0:
        raise ValueError("top-k must be >= 0")
    if args.max_pass_rate_drop < 0:
        raise ValueError("max-pass-rate-drop must be >= 0")
    if args.max_check_pass_rate_drop < 0:
        raise ValueError("max-check-pass-rate-drop must be >= 0")
    if args.max_avg_case_score_drop < 0:
        raise ValueError("max-avg-case-score-drop must be >= 0")

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_name = str(suite.get("name", "prompt_suite"))
    cases = list(suite.get("cases", []))
    if not cases:
        raise ValueError(f"suite has no cases: {suite_path}")

    runtime = _load_runtime(checkpoint_path, args.device)
    started_at = datetime.now(timezone.utc)

    case_reports: list[dict[str, Any]] = []
    passed_cases = 0
    check_count = 0
    check_passed = 0

    for idx, case in enumerate(cases):
        case_id = str(case.get("id", f"case_{idx:03d}"))
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"case missing prompt: {case_id}")
        category = str(case.get("category", "uncategorized"))
        checks = dict(case.get("checks", {}))

        seed = args.seed + idx * args.seed_stride
        gen = _generate(
            runtime,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_on_eos=True,
            seed=seed,
        )
        check_results, case_pass = _run_checks(gen["completion"], checks)
        if case_pass:
            passed_cases += 1
        for row in check_results:
            check_count += 1
            if bool(row["pass"]):
                check_passed += 1

        case_score = (
            sum(1 for row in check_results if bool(row["pass"])) / len(check_results)
            if check_results
            else 1.0
        )
        report_row = {
            "id": case_id,
            "category": category,
            "seed": seed,
            "pass": case_pass,
            "score": case_score,
            "token_count": gen["token_count"],
            "checks": check_results,
            "prompt": prompt,
            "completion": gen["completion"],
        }
        if args.keep_full_text:
            report_row["full_text"] = gen["full_text"]
        case_reports.append(report_row)
        print(f"case={case_id} pass={int(case_pass)} score={case_score:.3f}")

    finished_at = datetime.now(timezone.utc)
    pass_rate = passed_cases / len(case_reports)
    check_pass_rate = (check_passed / check_count) if check_count else 1.0
    avg_case_score = sum(float(row["score"]) for row in case_reports) / len(case_reports)

    report = {
        "suite_name": suite_name,
        "suite_path": str(suite_path),
        "suite_description": suite.get("description", ""),
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_step": runtime.step,
        "tokenizer_path": str(runtime.tokenizer_path),
        "device": str(runtime.device),
        "params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "seed_stride": args.seed_stride,
        },
        "summary": {
            "cases_total": len(case_reports),
            "cases_passed": passed_cases,
            "pass_rate": pass_rate,
            "checks_total": check_count,
            "checks_passed": check_passed,
            "check_pass_rate": check_pass_rate,
            "avg_case_score": avg_case_score,
        },
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "cases": case_reports,
    }
    current_summary = summary_from_report(report)

    baseline_report_path = Path(args.baseline_report) if args.baseline_report else None
    baseline_report_obj: dict[str, Any] | None = None
    if baseline_report_path is not None:
        if not baseline_report_path.exists():
            raise FileNotFoundError(f"baseline report not found: {baseline_report_path}")
        baseline_report_obj = json.loads(baseline_report_path.read_text(encoding="utf-8"))
        baseline_summary = summary_from_report(baseline_report_obj)
        deltas = compare_summaries(current_summary, baseline_summary)
        regression = {
            "baseline_report": str(baseline_report_path),
            "allowed_drop": {
                "pass_rate": args.max_pass_rate_drop,
                "check_pass_rate": args.max_check_pass_rate_drop,
                "avg_case_score": args.max_avg_case_score_drop,
            },
            "deltas": deltas,
            "pass": (
                deltas["pass_rate_delta"] >= -args.max_pass_rate_drop
                and deltas["check_pass_rate_delta"] >= -args.max_check_pass_rate_drop
                and deltas["avg_case_score_delta"] >= -args.max_avg_case_score_drop
            ),
        }
        report["regression"] = regression

    if args.promotion_policy:
        policy_path = Path(args.promotion_policy)
        if not policy_path.exists():
            raise FileNotFoundError(f"promotion policy not found: {policy_path}")
        policy_obj = json.loads(policy_path.read_text(encoding="utf-8"))
        if not isinstance(policy_obj, dict):
            raise ValueError("promotion policy must be a JSON object")
        baseline_summary_for_policy = (
            summary_from_report(baseline_report_obj)
            if baseline_report_obj is not None
            else None
        )
        promotion = evaluate_promotion_policy(
            current=current_summary,
            baseline=baseline_summary_for_policy,
            policy=policy_obj,
        )
        promotion["policy_path"] = str(policy_path)
        report["promotion"] = promotion

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_name = f"{suite_name}_{checkpoint_path.stem}_{timestamp}.json"
        output_path = Path("artifacts/reports/evals") / out_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={output_path}")
    print(
        "summary "
        f"cases_passed={passed_cases}/{len(case_reports)} "
        f"pass_rate={pass_rate:.3f} checks_passed={check_passed}/{check_count} "
        f"check_pass_rate={check_pass_rate:.3f} avg_case_score={avg_case_score:.3f}"
    )
    if "regression" in report:
        regression_obj = report["regression"]
        assert isinstance(regression_obj, dict)
        deltas = regression_obj["deltas"]
        assert isinstance(deltas, dict)
        print(
            "regression "
            f"pass={int(bool(regression_obj['pass']))} "
            f"pass_rate_delta={float(deltas['pass_rate_delta']):+.4f} "
            f"check_pass_rate_delta={float(deltas['check_pass_rate_delta']):+.4f} "
            f"avg_case_score_delta={float(deltas['avg_case_score_delta']):+.4f}"
        )
    if "promotion" in report:
        promotion_obj = report["promotion"]
        assert isinstance(promotion_obj, dict)
        failed_checks = promotion_obj.get("failed_checks", [])
        print(
            "promotion "
            f"pass={int(bool(promotion_obj.get('promoted', False)))} "
            f"failed_checks={','.join(str(x) for x in failed_checks) if failed_checks else 'none'}"
        )

    if args.fail_below_pass_rate is not None and pass_rate < args.fail_below_pass_rate:
        print(
            f"status=fail threshold={args.fail_below_pass_rate:.3f} observed_pass_rate={pass_rate:.3f}"
        )
        return 1
    if args.fail_on_regression and "regression" in report:
        regression_obj = report["regression"]
        assert isinstance(regression_obj, dict)
        if not bool(regression_obj.get("pass", False)):
            print("status=fail reason=regression_check_failed")
            return 1
    if args.fail_on_no_promotion and "promotion" in report:
        promotion_obj = report["promotion"]
        assert isinstance(promotion_obj, dict)
        if not bool(promotion_obj.get("promoted", False)):
            print("status=fail reason=promotion_policy_failed")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
