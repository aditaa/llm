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

This installs extras from `.[train,notebook]` (for example `torch`, `tiktoken`, `matplotlib`, `pandas`, `jupyterlab`).

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
