# Setup and Tooling

## System Requirements
- Ubuntu/Debian server environment
- Python 3.10+
- Git + SSH access to GitHub
- Optional GPU for training (NVIDIA + CUDA)

## Bootstrap
```bash
bash scripts/install_server_system.sh
bash scripts/bootstrap_dev.sh
bash scripts/bootstrap_train.sh
bash scripts/doctor.sh
```

## Common Development Commands
```bash
make test
make lint
make format
make typecheck
make smoke
make verify-shards
make train
make generate
make train-tokenizer-global
make shard-corpus-batch
make sync-warm
make hydrate-warm
make offload-zim
```

`make sync-warm` now includes raw/training inputs (`data/raw_zim`, `data/fineweb`,
`data/cleaned`, `data/extracted`) and training artifacts (`data/shards`,
`data/shards_global`, `artifacts/tokenizer`, `artifacts/checkpoints`, `artifacts/reports`).

## CI/CD
- `CI` workflow runs lint, typecheck, tests, and smoke checks.
- `Wiki Sync` workflow publishes docs in `wiki/` to the GitHub Wiki.
- `Dependabot` (`.github/dependabot.yml`) opens weekly dependency update PRs.
- Recommended branch protection: require `CI Gate` before merge to `main`.

## Repository Structure
- `src/llm/`: tokenizer, data, sharding, training, CLI, model modules
- `tests/`: unit test coverage for core data/tokenizer/sharding paths
- `docs/`: architecture and server notes
- `information/`: reference notes and external source material
- `scripts/`: setup and operational scripts

## RTX 5070 Training Profile
- Tuning notes: `docs/RTX5070_TUNING.md`
- Saved configs: `configs/train/rtx5070/`
- Recommended launcher:
```bash
bash scripts/train_rtx5070_fineweb_bpe_v1_big.sh
```
- Training defaults now include warmup+cosine LR and fixed held-out eval batches.
- For VRAM pressure, increase effective batch with `--grad-accum-steps`.
- For release bundles, use `--include-safetensors` in HF prepare script.

## Wiki Publishing
Wiki pages are source-controlled in `wiki/` and published with:

```bash
make publish-wiki
```

This command pushes all `wiki/*.md` pages to `https://github.com/aditaa/llm/wiki`.
