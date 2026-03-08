# Repository Guidelines

## Project Structure & Module Organization
The codebase is a Python LLM-from-scratch scaffold.
- `src/llm/`: core package (`tokenizer.py`, `data.py`, `sharding.py`, `cli.py`, `model.py`)
- `tests/`: unit tests (tokenizer + data pipeline + sharding coverage)
- `docs/`: architecture and roadmap docs
- `information/`: references, imported notes, and source material
- `data/`: local/intermediate corpora (gitignored except `data/README.md`)
- `artifacts/`: generated outputs such as vocab/checkpoints (gitignored)

Keep modules single-purpose and expand by domain (for example: `src/llm/training.py`, `src/llm/data.py`).

## Build, Test, and Development Commands
Use the `Makefile` as the source of truth:
- `make setup-dev`: create `.venv`, install dev deps, init submodules
- `make setup-train`: install training/notebook extras
- `make setup-infer`: install inference/deploy extras
- `make install-server-system`: install Ubuntu/Debian system packages
- `make doctor`: environment/tooling diagnostics
- `make test`: run `unittest` test suite
- `make lint`: run Ruff lint checks
- `make format`: run Black formatter
- `make typecheck`: run MyPy on `src/`
- `make smoke`: run a minimal CLI smoke test
- `make verify-shards`: usage helper for shard integrity verification
- `make train`: usage helper for baseline GPT training
- `make generate`: usage helper for checkpoint text generation
- `make eval-checkpoint`: usage helper for standardized checkpoint prompt-suite eval
- `make train-tokenizer-global`: usage helper for shared tokenizer training
- `make corpus-quality-report`: usage helper for corpus quality scan
- `make clean-corpus-batch`: usage helper for batch corpus cleanup
- `make dataset-risk-report`: usage helper for heuristic dataset risk audit
- `make pull-hf-rows`: usage helper for bounded Hugging Face rows pulls
- `make fineweb-parquet-to-shards`: usage helper for direct FineWeb parquet -> tokenizer -> shard conversion
- `make stage-fineweb-from-warm`: usage helper for staging FineWeb parquet chunks from warm to hot
- `make fineweb-stage-shard-loop`: usage helper for rolling warm->hot stage + shard + verify + sync + purge
- `make fineweb-hot-queue`: usage helper for hot parquet queue-oriented stage + shard flow
- `make lr-sweep-350bt`: usage helper for RTX 5070 LR sweep on staged 350BT shards (`2e-4..4e-4`, ctx 512)
- `make train-350bt-v2`: usage helper for the 350BT long-run launcher profile
- `make train-supervisor-350bt`: usage helper for auto-resume chunked training that refreshes manifest set between cycles
- `make pipeline-eta`: usage helper for combined download + sharding + training ETA/status reporting
- `make pipeline-live`: usage helper for a live terminal pipeline dashboard
- `make shard-corpus-batch`: usage helper for batch sharding with a shared tokenizer
- `make hf-download-resumable`: usage helper for self-healing Hugging Face resume-download worker
- `make hf-prepare-publish`: usage helper for Hugging Face release bundle/publish
- `make hf-download-model`: usage helper for full Hugging Face model snapshot download
- `make serve-openai`: usage helper for local OpenAI-compatible serving

Server setup reference:
`docs/SERVER_SETUP.md`

CI/CD workflows:
- `.github/workflows/ci.yml`: lint, typecheck, tests, smoke, and gate job
- `.github/workflows/wiki-sync.yml`: publishes wiki pages on `main` doc changes
- `.github/dependabot.yml`: weekly dependency update PRs for `pip`, `requirements/`, and GitHub Actions

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, UTF-8 files
- Max line length: 100 (Black/Ruff configured in `pyproject.toml`)
- Use `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_CASE` for constants
- Add type hints for public functions; MyPy is configured with `disallow_untyped_defs = true`

## Testing Guidelines
- Framework: `unittest` (`tests/test_*.py`)
- Mirror source layout when adding tests (example: `src/llm/data.py` -> `tests/test_data.py`)
- Cover edge cases and failure modes, not only happy paths
- Run `make test` locally before each commit

## Commit & Pull Request Guidelines
Follow concise, imperative commit subjects (<= 72 chars), for example:
- `Add tokenizer save/load round-trip tests`
- `Implement corpus stats CLI command`

PRs should include:
- What changed and why
- How it was validated (`make test`, `make lint`, sample output if relevant)
- Linked issue/ticket when applicable

Keep PR scope narrow; split refactors and features into separate PRs.
`main` should be protected with required status check `CI Gate`.

## Security & Configuration Tips
- Never commit secrets or credentials
- Never commit raw dataset dumps or `.zim` archives
- Never commit copyrighted book/PDF source files
- Keep generated files in `artifacts/` and out of git history
- Prefer environment variables for machine-specific settings
- Run all processing in local hot workspace (`./data`, `./artifacts`)
- Use warm storage at `/mnt/ceph/llm/data` as cache/backup
- For first-pass talking-only selection + hot/warm rebalance, run:
  `bash scripts/first_pass_zim_profile.sh --move-excluded`
- Run shard integrity verification before training (`llm.cli verify-shards`)
- `extract-zim-text` now falls back to suggestion-index paths when fulltext search has zero matches
- If extraction still returns `written_articles=0`, retry with a lower `--min-chars` (for example `20`)
- For ZIMs without fulltext index, generate a paths list from suggestion/title index and run `extract-zim-text --paths-file ...`
- `llm.cli train` requires a tokenizer-compatible shard set (same tokenizer mapping across all selected manifests)
- Shard manifests now record `tokenizer_hash` + `tokenizer_contract_hash`; do not mix manifests with mismatched values
- Preferred multi-dataset flow: `train-tokenizer-global` -> `shard-corpus-batch` -> `train`
- Preferred pre-tokenization flow: `corpus-quality-report` -> `clean-corpus-batch` -> `train-tokenizer-global`
- Run `dataset-risk-report` on cleaned corpora before tokenizer training and manually review flagged slices
- Prefer `clean-corpus-batch --en-only` for first-pass talking runs
- For talking-only runs, keep `clean-corpus-batch` code-like filtering enabled (default)
- Cleanup defaults also enforce min words, URL density, symbol density, and repetitive-token filters
- Use `bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data` to copy local artifacts to warm storage
- `sync_warm_storage.sh` covers raw + training data: `data/raw_zim`, `data/fineweb`, `data/cleaned`, `data/extracted`, `data/shards`, `data/shards_global`, `artifacts/tokenizer`, `artifacts/checkpoints`, `artifacts/reports`
- Use `bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120` for continuous hot->warm raw ZIM offload
- Use `bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data` to restore local artifacts from warm storage
- For bounded external pulls (for example FineWeb samples), use `python3 scripts/pull_hf_rows.py` and write to warm storage first
- For long-running Hugging Face parquet pulls, use `scripts/hf_download_resumable.sh` instead of one-shot `hf download` (prefer `--enable-hf-transfer`, `--max-workers 6`, `--skip-dry-run`, and `--attempt-timeout-seconds` for 350BT-scale pulls)
- For parquet-based FineWeb workflows, use `scripts/stage_fineweb_from_warm.sh` to copy bounded warm chunks into hot storage
- For long-running 350BT ingestion on limited hot disk, use `scripts/fineweb_stage_shard_loop.sh` for staged processing and automatic hot-space reclaim
- Use `fineweb_stage_shard_loop.sh --hot-queue-min-files <N>` to keep a bounded hot parquet queue and reduce sharder copy stalls
- Keep stage-loop OOM retry enabled (default) so shard builds back off to smaller `--batch-size` automatically
- For continuously growing shard sets, use `scripts/train_supervisor_rtx5070_350bt.sh` so each resumed chunk re-reads new manifests before training continues
- Supervisor writes chunk trends to `artifacts/reports/train_supervisor_350bt/train_trend.tsv` and post-chunk eval trends to `artifacts/reports/train_supervisor_350bt/eval_trend.tsv`
- Use `scripts/pipeline_eta_report.py --loop` for combined ETA snapshots in `artifacts/reports/pipeline_status.{json,txt}` (includes `top`, `free -h`, `nvidia-smi`, and `df -h` captures)
- Use `scripts/pipeline_live_view.py --refresh-seconds 5` for a live-only terminal monitor (system + pipeline task status, no report writes; add `--no-alt-screen` if needed)
- For checkpoint regression tracking, run `scripts/eval_checkpoint_prompts.py` with `configs/eval/standard_prompt_suite_v1.json` and archive reports in `artifacts/reports/evals/`
- For FineWeb-first training runs, build shards directly with `PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py --input-dir data/fineweb/sample-350BT --output-dir data/shards_global/fineweb-global-bpe-v1 --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json --bpe-vocab-size 32000 --field text`
- FineWeb-only baseline flow: `fineweb_parquet_to_shards -> verify-shards -> train`
- For incremental FineWeb adds, freeze tokenizer on phase1 and build later phases with `--tokenizer-in` plus `--files-list`; resume training from `last.pt` with same `--shards-path` root
- On this 20-core server, use 15 parallel streams for split shard-build runs
- For CUDA training throughput, prefer `llm.cli train --precision auto` (disable TF32 only if needed with `--no-tf32`)
- If GPU utilization stays bursty, try `llm.cli train --compile-model --compile-mode reduce-overhead`
- Default training architecture is `gpt_rope_rmsnorm_swiglu_v1`; use legacy profile only for old checkpoint compatibility
- Prefer `llm.cli train --lr-schedule cosine --lr-warmup-steps <N>` for stable first-pass runs
- Use `--grad-accum-steps` when VRAM is tight and you need higher effective batch
- Keep held-out eval batches frozen (default) and enable regression gating with `--fail-on-eval-regression`
- Optimizer uses no-weight-decay groups for norms/biases/embeddings by default
- For deploy bundles, export weights-only safetensors via `llm.cli train --export-safetensors` or `hf_prepare_and_publish_model.py --include-safetensors`
- RTX 5070 tuned training profiles live in `configs/train/rtx5070/`; preferred 350BT launchers: `bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh` then `bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh`
- Version extracted/tokenized/sharded outputs with the ZIM date stamp (for example `serverfault_2025-08`)
- Keep raw ZIM archives in `/mnt/ceph/llm/data/raw_zim/`
- For portable model release + offline server deploy, follow `docs/HF_RELEASE_AND_DEPLOY.md`
- On restricted-network deploy hosts, download full model snapshots locally with `hf-download-model` before serving

## Reference Material Workflow
- Store reusable project references in `information/`
- Start with `information/README.md` for curated external links
- Keep `information/raschka-reference-notes.md` updated when Raschka source material informs implementation
- Track execution progress in `information/raschka-implementation-checklist.md`
- Use `information/external/LLMs-from-scratch` (git submodule) for direct code reference
- Refresh submodule when needed with: `git submodule update --remote information/external/LLMs-from-scratch`
- When adding a new source, include a short note on why it matters to this codebase

## Wiki Maintenance
- Keep repository wiki pages source-controlled in `wiki/`
- Publish wiki updates with: `bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git`
- Keep `wiki/Dataset-Registry.md` updated with every newly approved source and intended usage
- When docs change, update `README.md`, `AGENTS.md`, and relevant `wiki/*.md` pages in the same PR
