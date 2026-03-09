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
make render-eval-dashboard
make package-inference-bundle
make train-tokenizer-global
make shard-corpus-batch
make install-systemd-services
make sync-warm
make hydrate-warm
make offload-zim
make checkpoint-offload-prune
make set-swappiness
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
- For long-context continuation, use `bash scripts/train_rtx5070_fineweb_350bt_bpe_v2_ctx1024.sh`.
- For release bundles, use `scripts/package_inference_bundle.py` or `--include-safetensors` in HF prepare script.

## systemd Services
Install reboot-safe workers:
```bash
make install-systemd-services
```
Units installed from `deploy/systemd/`:
- `llm-train-supervisor.service`
- `llm-fineweb-stage-shard-loop.service`
- `llm-fineweb-stage-shard-watchdog.service`
- `llm-hf-download-watchdog.service` (optional)
- `llm-checkpoint-offload-prune.service`
- `llm-checkpoint-offload-prune.timer`
- `llm-vm-swappiness.service`

Optional prefetch service (only if you want separate prefetch in addition to stage-loop staging):
- `llm-fineweb-prefetch.service`
- install with: `bash scripts/install_systemd_services.sh --install-watchdog --install-prefetch`

`deploy/systemd/llm.env.example` includes tuned loop args for this host profile
(`--hot-queue-min-files 10`, `--stage-copy-jobs 4`, `--stage-min-free-gib 80`,
`--auto-tune-shard-jobs`, `--sync-background`, `--shard-size-tokens 20000000`).
Override `LLM_STAGE_SHARD_LOOP_ARGS` in `/etc/llm/llm.env` if you want a different policy.
Set `LLM_CHECKPOINT_OFFLOAD_ARGS` for checkpoint warm-sync/prune policy and
`LLM_SWAPPINESS=10` for VM tuning on boot.

## Wiki Publishing
Wiki pages are source-controlled in `wiki/` and published with:

```bash
make publish-wiki
```

This command pushes all `wiki/*.md` pages to `https://github.com/aditaa/llm/wiki`.
