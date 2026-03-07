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
  --suite configs/eval/standard_prompt_suite_v1.json
```

This writes scored JSON reports to `artifacts/reports/evals/` for run-to-run comparison.

## Wiki Maintenance
When docs change in repo:
1. Update pages in `wiki/`.
2. Run `make publish-wiki`.
3. Commit repo doc changes and push.
