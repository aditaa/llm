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
make sync-warm
make hydrate-warm
```

## CI/CD
- `CI` workflow runs lint, typecheck, tests, and smoke checks.
- `Wiki Sync` workflow publishes docs in `wiki/` to the GitHub Wiki.
- Recommended branch protection: require `CI Gate` before merge to `main`.

## Repository Structure
- `src/llm/`: tokenizer, data, sharding, CLI, model modules
- `tests/`: unit test coverage for core data/tokenizer/sharding paths
- `docs/`: architecture and server notes
- `information/`: reference notes and external source material
- `scripts/`: setup and operational scripts

## Wiki Publishing
Wiki pages are source-controlled in `wiki/` and published with:

```bash
make publish-wiki
```

This command pushes all `wiki/*.md` pages to `https://github.com/aditaa/llm/wiki`.
