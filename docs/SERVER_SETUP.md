# Server Setup

This document defines the required tooling for running this repository on an Ubuntu/Debian server.

## 1) System Packages
Install system dependencies listed in:

`requirements/server-system-ubuntu.txt`

Install command:

```bash
bash scripts/install_server_system.sh
```

## 2) Python Environment (Dev)
Create a virtual environment and install developer dependencies:

```bash
bash scripts/bootstrap_dev.sh
```

This installs:
- package in editable mode
- lint/type tools (`ruff`, `black`, `mypy`)
- initializes git submodules recursively

## 3) Training Extras
Install training/notebook dependencies when needed:

```bash
bash scripts/bootstrap_train.sh
```

This installs extras from `.[train,notebook,data]` (for example `torch`, `tiktoken`, `matplotlib`, `pandas`, `jupyterlab`).
This also installs `libzim` for ZIM corpus extraction.

## 4) Environment Verification
Run diagnostics:

```bash
bash scripts/doctor.sh
```

Or via Make:

```bash
make doctor
```

## 5) Daily Commands
```bash
source .venv/bin/activate
make test
make smoke
make lint
make typecheck
```

## 6) ZIM Data Location
- Keep raw IIAB `.zim` files outside Git, on server storage (for example `/data/iiab/zim/`).
- Extract corpus text into `data/extracted/` in this repo.
- `data/` is gitignored except `data/README.md`.

## 7) Warm Storage (Recommended)
If a Ceph/NFS mount is available (for example `/mnt/ceph/llm/data`), use it for large extracted and sharded artifacts.

Suggested directories:
- `/mnt/ceph/llm/data/raw_zim/`
- `/mnt/ceph/llm/data/extracted/`
- `/mnt/ceph/llm/data/shards/`
- `/mnt/ceph/llm/data/tokenizer/`

Use a date-based version tag from the ZIM filename for all derived artifacts.
Example: `serverfault.com_en_all_2025-08.zim` -> `serverfault_2025-08`.

Sync local artifacts when needed:

```bash
bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data
```
