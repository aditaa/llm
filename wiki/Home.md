# LLM From Scratch Wiki

This wiki is the operational documentation for the `llm` repository.

## What This Project Does
- Builds a decoder-only LLM stack from scratch.
- Focuses on reproducible data prep, tokenization, sharding, training, and evaluation.
- Uses Sebastian Raschka's materials as implementation references.

## Start Here
- [Setup and Tooling](Setup-and-Tooling)
- [Data Pipeline and Versioning](Data-Pipeline-and-Versioning)
- [Development Workflow](Development-Workflow)
- [Architecture and Roadmap](Architecture-and-Roadmap)
- [References](References)

## Current Status
- ZIM extraction pipeline implemented.
- Tokenizer training pipeline implemented.
- Corpus sharding pipeline implemented.
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
