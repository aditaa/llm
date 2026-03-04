# LLM From Scratch

This repository is the base for building a decoder-only language model from first principles, with clear milestones for data prep, tokenization, model training, evaluation, and inference.

## Project Goals
- Build a minimal but production-style training stack incrementally.
- Keep each subsystem testable (`tokenizer`, `data`, `model`, `training`, `evaluation`).
- Favor reproducible experiments through explicit configs and scripts.

## Repository Layout
- `src/llm/`: core Python package
- `tests/`: unit tests
- `docs/`: architecture and roadmap notes
- `information/`: reference material and external links for project guidance
- `artifacts/`: local outputs (vocab, checkpoints, logs; gitignored)
- `Makefile`: common developer commands

## Quick Start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Common Commands
```bash
make test        # run unit tests
make lint        # run Ruff checks
make format      # run Black formatter
make smoke       # tiny CLI smoke check
```

## Current Capabilities
- Text stats CLI for quick corpus sanity checks.
- Basic character-level tokenizer with train/save/load.
- Unit tests for tokenizer round-trips and unknown token behavior.

## Next Milestones
1. Add dataset batching and sequence packing.
2. Implement GPT-style transformer blocks in `src/llm/model.py`.
3. Add a first training loop and checkpointing.
4. Add validation metrics (loss, perplexity) and text generation.

## References
- Internal reference index: `information/README.md`
- Working notes from loaded PDF + external references: `information/raschka-reference-notes.md`
- Implementation checklist from those references: `information/raschka-implementation-checklist.md`
- Sebastian Raschka article: https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up
- Raschka repository: https://github.com/rasbt/LLMs-from-scratch
- Local checkout (submodule): `information/external/LLMs-from-scratch`

## Reference Repo Sync
```bash
git submodule update --init --recursive
git submodule update --remote information/external/LLMs-from-scratch
```
Use the first command after clone; use the second to pull newer upstream reference commits.
