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
- `make train-tokenizer-global`: usage helper for shared tokenizer training
- `make corpus-quality-report`: usage helper for corpus quality scan
- `make clean-corpus-batch`: usage helper for batch corpus cleanup
- `make dataset-risk-report`: usage helper for heuristic dataset risk audit
- `make pull-hf-rows`: usage helper for bounded Hugging Face rows pulls
- `make fineweb-parquet-to-shards`: usage helper for direct FineWeb parquet -> tokenizer -> shard conversion
- `make stage-fineweb-from-warm`: usage helper for staging FineWeb parquet chunks from warm to hot
- `make shard-corpus-batch`: usage helper for batch sharding with a shared tokenizer
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
- Preferred multi-dataset flow: `train-tokenizer-global` -> `shard-corpus-batch` -> `train`
- Preferred pre-tokenization flow: `corpus-quality-report` -> `clean-corpus-batch` -> `train-tokenizer-global`
- Run `dataset-risk-report` on cleaned corpora before tokenizer training and manually review flagged slices
- For English-only runs, enable `clean-corpus-batch --en-only` before tokenizer training
- For talking-only runs, keep `clean-corpus-batch` code-like filtering enabled (default)
- Use `bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data` to copy local artifacts to warm storage
- `sync_warm_storage.sh` covers raw + training data: `data/raw_zim`, `data/fineweb`, `data/cleaned`, `data/extracted`, `data/shards`, `data/shards_global`, `artifacts/tokenizer`, `artifacts/checkpoints`, `artifacts/reports`
- Use `bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120` for continuous hot->warm raw ZIM offload
- Use `bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data` to restore local artifacts from warm storage
- For bounded external pulls (for example FineWeb samples), use `python3 scripts/pull_hf_rows.py` and write to warm storage first
- For parquet-based FineWeb workflows, use `scripts/stage_fineweb_from_warm.sh` to copy bounded warm chunks into hot storage
- For FineWeb-first training runs, build shards directly with `PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py --input-dir data/fineweb/sample-10BT --output-dir data/shards_global/fineweb-s10bt-global-char-v1 --tokenizer-out artifacts/tokenizer/fineweb-s10bt-global-char-v1.json --field text`
- FineWeb-only baseline flow: `fineweb_parquet_to_shards -> verify-shards -> train`
- For incremental FineWeb adds, freeze tokenizer on phase1 and build later phases with `--tokenizer-in` plus `--files-list`; resume training from `last.pt` with same `--shards-path` root
- On this 20-core server, use 15 parallel streams for split shard-build runs
- For CUDA training throughput, prefer `llm.cli train --precision auto` (disable TF32 only if needed with `--no-tf32`)
- If GPU utilization stays bursty, try `llm.cli train --compile-model --compile-mode reduce-overhead`
- RTX 5070 tuned training profiles live in `configs/train/rtx5070/`; default launcher: `bash scripts/train_rtx5070_fineweb_v2_big.sh`
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
