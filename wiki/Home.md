# LLM From Scratch Wiki

Operational docs for the `llm` repository, focused on reproducible data prep, tokenizer/sharding pipelines, and GPU training.

## About
- Decoder-only LLM project built from scratch.
- Emphasis on engineering reliability: CI gate, integrity checks, and versioned data flows.
- Hot/warm storage model for large corpora (`./data` + `/mnt/ceph/llm/data`).

## What This Project Does
- Builds a decoder-only LLM stack from scratch.
- Focuses on reproducible data prep, tokenization, sharding, training, and evaluation.
- Uses Sebastian Raschka's materials as implementation references.

## Start Here
- [Setup and Tooling](Setup-and-Tooling)
- [Data Pipeline and Versioning](Data-Pipeline-and-Versioning)
- [Development Workflow](Development-Workflow)
- [Architecture and Roadmap](Architecture-and-Roadmap)
- [Release and Deployment](Release-and-Deployment)
- [References](References)

## Current Status
- ZIM extraction pipeline implemented.
- Tokenizer training pipeline implemented.
- Corpus sharding pipeline implemented.
- Shared tokenizer + batch sharding workflow implemented.
- Baseline GPT training + checkpoint/resume implemented.
- Hot-local processing + warm-storage cache workflow implemented.
- CI/CD workflows implemented for checks and wiki publishing.

## Core Commands
```bash
make setup-dev
make test
make lint
make typecheck
make train
```

For server-specific setup and storage guidance, see [Setup and Tooling](Setup-and-Tooling) and [Data Pipeline and Versioning](Data-Pipeline-and-Versioning).
