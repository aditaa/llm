# Development Workflow

## Quality Gate
Run these before every push:

```bash
make lint
make typecheck
make test
```

CI enforces these checks in GitHub Actions, with `CI Gate` as the merge gate for `main`.

## Commit Conventions
- Use concise imperative messages, <=72 chars.
- Keep commits scoped (one feature/fix per commit).
- Include docs updates (`README.md`, `AGENTS.md`, wiki pages) with behavior changes.

Examples:
- `Add corpus sharding pipeline and CLI command`
- `Add Ceph warm-storage sync workflow and versioning docs`

## Pull Requests
Each PR should include:
- What changed and why
- Validation evidence (commands run + key results)
- Any operational impact (data paths, storage use, migration)

## Data Safety Rules
- Never commit raw datasets or `.zim` files to Git.
- Keep generated artifacts in gitignored paths.
- Prefer durable storage for large data (`/mnt/ceph/llm/data`).

## Daily Operations
```bash
source .venv/bin/activate
make test
make smoke
```

## Checkpoint Eval Baseline
Run a fixed prompt-suite eval after each major training run:

```bash
PYTHONPATH=src .venv/bin/python scripts/eval_checkpoint_prompts.py \
  --checkpoint artifacts/checkpoints/<run>/last.pt \
  --suite configs/eval/standard_prompt_suite_v2.json \
  --baseline-report artifacts/reports/evals/<previous_report>.json \
  --promotion-policy configs/eval/promotion_policy_v1.json \
  --fail-on-regression
```

This writes scored JSON reports to `artifacts/reports/evals/` for run-to-run comparison,
including regression deltas and promotion verdict fields.

## EMA and Checkpoint Smoothing
For longer runs, enable EMA in training and optionally evaluate/generate from EMA weights:

```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/<run> \
  --ema-decay 0.999 \
  --ema-start-step 1000
```

Merge adjacent checkpoints into one smoother snapshot:

```bash
PYTHONPATH=src .venv/bin/python -m llm.cli average-checkpoints \
  --checkpoint artifacts/checkpoints/<run>/ckpt_step_0001000.pt \
  --checkpoint artifacts/checkpoints/<run>/ckpt_step_0002000.pt \
  --output artifacts/checkpoints/<run>/avg_last2.pt \
  --state-key model_state
```

## Wiki Maintenance
When docs change in repo:
1. Update pages in `wiki/`.
2. Run `make publish-wiki`.
3. Commit repo doc changes and push.
