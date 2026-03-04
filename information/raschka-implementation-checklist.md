# Raschka-Informed Implementation Checklist

Last updated: 2026-03-04

Use this as the execution tracker for turning the Raschka references into code in this repository.

## 0) Baseline and Tooling
- [x] Add Raschka references and summary notes
  - [x] `information/raschka-reference-notes.md`
  - [x] `information/external/LLMs-from-scratch` submodule
- [x] Bootstrapped Python package and test scaffold
  - [x] `src/llm/`
  - [x] `tests/`
  - [x] `Makefile`, `pyproject.toml`
- [ ] Install local dev dependencies
  - Command: `pip install -e ".[dev]"`
  - Verify: `make lint`, `make typecheck`, `make test`

## 1) Data Pipeline (Chapter 2 Track)
- [ ] Create `src/llm/data.py`
  - [ ] Dataset class for token sequence windows
  - [ ] Next-token input/target shifting
  - [ ] Train/val split helper
  - [ ] Batch collation helper
- [ ] Extend CLI in `src/llm/cli.py`
  - [ ] Add command to build tokenized dataset artifacts
  - [ ] Add command to preview batch shapes/stats
- [ ] Add tests in `tests/test_data.py`
  - [ ] Windowing edge cases
  - [ ] Shift correctness (`x[t] -> y[t+1]`)
  - [ ] Reproducible split behavior with seed

## 2) Model Core (Chapters 3-4 Track)
- [ ] Expand `src/llm/model.py`
  - [ ] `MultiHeadAttention` with causal mask
  - [ ] Feed-forward MLP block
  - [ ] Transformer block with residual + layer norm
  - [ ] GPT model (token embedding + positional embedding + blocks + LM head)
- [ ] Add tests in `tests/test_model.py`
  - [ ] Shape invariants (`[B,T,V]` logits)
  - [ ] Causal masking no-lookahead check
  - [ ] Forward pass determinism under fixed seed

## 3) Training + Generation (Chapter 5 Track)
- [ ] Add `src/llm/training.py`
  - [ ] Train step + eval step
  - [ ] Loss logging for train/validation
  - [ ] Checkpoint save/load
  - [ ] Resume training from checkpoint
- [ ] Add `src/llm/generate.py`
  - [ ] Greedy decoding
  - [ ] Top-k sampling
  - [ ] Top-p sampling
- [ ] Add CLI wiring in `src/llm/cli.py`
  - [ ] `train` command
  - [ ] `generate` command
- [ ] Add tests
  - [ ] `tests/test_training.py` (checkpoint roundtrip)
  - [ ] `tests/test_generate.py` (sampling behavior sanity)

## 4) Finetuning (Chapters 6-7 Track)
- [ ] Add classification finetuning module
  - [ ] File: `src/llm/finetune_classification.py`
  - [ ] Dataset adapter + labels
  - [ ] Metrics: accuracy + confusion summary
- [ ] Add instruction finetuning module
  - [ ] File: `src/llm/finetune_instruction.py`
  - [ ] Instruction dataset schema checks
  - [ ] SFT train loop
- [ ] Add evaluation module
  - [ ] File: `src/llm/eval.py`
  - [ ] Response generation + scoring hooks
- [ ] Add tests
  - [ ] `tests/test_finetune_classification.py`
  - [ ] `tests/test_finetune_instruction.py`

## 5) Experiment and Config Discipline
- [ ] Add config directory
  - [ ] `configs/pretrain/*.yaml`
  - [ ] `configs/finetune/*.yaml`
- [ ] Add artifact conventions
  - [ ] Checkpoints in `artifacts/checkpoints/`
  - [ ] Metrics in `artifacts/metrics/`
  - [ ] Generated samples in `artifacts/samples/`
- [ ] Add run manifest output
  - [ ] Git commit hash
  - [ ] Config snapshot
  - [ ] Seed and environment info

## 6) Quality Gates
- [ ] Ensure Make targets are complete
  - [ ] `make test`
  - [ ] `make lint`
  - [ ] `make format`
  - [ ] `make typecheck`
  - [ ] `make smoke`
- [ ] Add CI workflow
  - [ ] Run lint + tests + typecheck on PR
  - [ ] Cache dependencies for faster checks

## 7) Documentation Sync (Required Every Feature)
- [ ] Update `README.md`
  - [ ] New commands
  - [ ] New module summaries
  - [ ] Example usage
- [ ] Update `AGENTS.md`
  - [ ] Structure changes
  - [ ] Workflow/convention changes
- [ ] Update `information/raschka-reference-notes.md`
  - [ ] Add new relevant upstream references used
  - [ ] Add mapping notes for implemented features

## 8) “Done” Criteria for Each Milestone
- [ ] Code exists in `src/llm/` for the targeted milestone
- [ ] Tests added and passing locally
- [ ] CLI command (or script entry point) exists and runs
- [ ] Docs (`README.md`, `AGENTS.md`) updated
- [ ] Changes committed and pushed

## 9) Upstream Reference Sync Checklist
- [ ] Pull latest reference repo commits
  - Command: `git submodule update --remote information/external/LLMs-from-scratch`
- [ ] Reconcile reference notes with upstream changes
  - File: `information/raschka-reference-notes.md`
- [ ] Commit submodule pointer update (if changed)
  - Include brief summary of what changed upstream
