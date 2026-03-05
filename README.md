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
- `requirements/`: system and Python dependency lists for server setup
- `scripts/`: bootstrap/install/doctor scripts
- `data/`: local/intermediate corpora (gitignored except `data/README.md`)
- `artifacts/`: local outputs (vocab, checkpoints, logs; gitignored)
- `Makefile`: common developer commands

## Quick Start
```bash
bash scripts/bootstrap_dev.sh
```

## Common Commands
```bash
make test        # run unit tests
make lint        # run Ruff checks
make format      # run Black formatter
make typecheck   # run MyPy
make smoke       # tiny CLI smoke check
make verify-shards # print shard integrity check usage
make doctor      # verify binaries and Python deps
```

## CI/CD
GitHub Actions workflows are defined in `.github/workflows/`:
- `ci.yml`: lint, typecheck, unit tests, smoke checks on pull requests and pushes to `main`
- `wiki-sync.yml`: publish `wiki/*.md` changes to the GitHub Wiki

Recommended branch protection for `main`:
- Require pull request before merging
- Require status checks: `CI Gate`
- Require branches to be up to date before merge

## Server Setup (Ubuntu/Debian)
1. Install system packages:
   `bash scripts/install_server_system.sh`
2. Bootstrap dev environment:
   `bash scripts/bootstrap_dev.sh`
3. Install training extras:
   `bash scripts/bootstrap_train.sh`
4. Run health check:
   `bash scripts/doctor.sh`

Detailed guide: `docs/SERVER_SETUP.md`

## ZIM Data Workflow (IIAB)
Keep raw `.zim` files on server storage (for example `/data/iiab/zim/`), not in Git.

1. Extract text corpus from ZIM:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli extract-zim-text \
  --input-zim /data/iiab/zim/wikipedia_en_all_maxi.zim \
  --output data/extracted/wiki_corpus.txt \
  --max-articles 50000 \
  --min-chars 200
```

2. Train tokenizer on extracted corpus:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer \
  --input data/extracted/wiki_corpus.txt \
  --output artifacts/tokenizer/vocab.json
```

3. Shard tokenized corpus for training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus \
  --input data/extracted/wiki_corpus.txt \
  --tokenizer artifacts/tokenizer/vocab.json \
  --output-dir data/shards/wiki_char \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```

4. Inspect corpus quickly:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli stats --input data/extracted/wiki_corpus.txt
```

5. Verify shard integrity before training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards \
  --raw-zim-dir data/raw_zim \
  --strict-source
```

## Warm Storage (Ceph Mount)
Use `./data` and `./artifacts` as the hot working set.
Use `/mnt/ceph/llm/data` as warm cache/backup for durability and overflow.

- Recommended mount layout:
  - `/mnt/ceph/llm/data/raw_zim/`
  - `/mnt/ceph/llm/data/extracted/`
  - `/mnt/ceph/llm/data/shards/`
  - `/mnt/ceph/llm/data/tokenizer/`
- Version datasets by ZIM date stamp:
  - ZIM: `serverfault.com_en_all_2025-08.zim`
  - Version tag: `serverfault_2025-08`
  - Raw ZIM: `/mnt/ceph/llm/data/raw_zim/serverfault.com_en_all_2025-08.zim`
  - Extracted text: `/mnt/ceph/llm/data/extracted/serverfault_2025-08.txt`
  - Tokenizer: `/mnt/ceph/llm/data/tokenizer/serverfault_2025-08-vocab.json`
  - Shards: `/mnt/ceph/llm/data/shards/serverfault_2025-08/`
- Default run model:
  - Process locally in `data/extracted`, `data/shards`, and `artifacts/tokenizer`.
  - Periodically sync to Ceph for backup/caching.
- Push local artifacts to warm storage:
```bash
bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data
```
- Pull artifacts back from warm storage to local hot workspace:
```bash
bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data
```

## Current Capabilities
- Text stats CLI for quick corpus sanity checks.
- Basic character-level tokenizer with train/save/load.
- Token-window data pipeline (`TokenWindowDataset`) for next-token training pairs.
- ZIM archive text extraction (`extract-zim-text`) for server-hosted `.zim` files.
- Corpus sharding (`shard-corpus`) into train/val token shard binaries + manifest.
- Unit tests for tokenizer round-trips and unknown token behavior.

## Next Milestones
1. Implement GPT-style transformer blocks in `src/llm/model.py`.
2. Add a first training loop and checkpointing.
3. Add validation metrics (loss, perplexity) and text generation.
4. Add finetuning flows for classification and instruction datasets.

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

## Wiki Documentation
Repository wiki pages are maintained from `wiki/*.md`.

Publish updates to GitHub wiki:
```bash
bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git
```

Preferred workflow:
1. Update `README.md` and `AGENTS.md` as needed.
2. Update matching pages in `wiki/`.
3. Publish wiki with `scripts/publish_wiki.sh`.
