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
make train       # print baseline training command usage
make generate    # print checkpoint text-generation command usage
make train-tokenizer-global # print shared-tokenizer command usage
make corpus-quality-report # print quality report command usage
make clean-corpus-batch # print batch cleanup command usage
make shard-corpus-batch # print shared-tokenizer batch sharding usage
make doctor      # verify binaries and Python deps
```

## CI/CD
GitHub Actions workflows are defined in `.github/workflows/`:
- `ci.yml`: lint, typecheck, unit tests, smoke checks on pull requests and pushes to `main`
- `wiki-sync.yml`: publish `wiki/*.md` changes to the GitHub Wiki
- Dependabot config: `.github/dependabot.yml` (weekly updates for `pip`, `requirements/`, and GitHub Actions)

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

For a first-pass talking-only dataset profile (English prose focus), generate include/exclude manifests:
```bash
bash scripts/first_pass_zim_profile.sh
```
To also move excluded local ZIMs from hot storage to warm storage:
```bash
bash scripts/first_pass_zim_profile.sh --move-excluded
```
This writes:
- `artifacts/reports/first_pass_include_targets.txt` (target profile, includes Gutenberg)
- `artifacts/reports/first_pass_include_zims.txt` (currently present and included)
- `artifacts/reports/first_pass_exclude_zims.txt` (currently present and excluded)

1. Extract text corpus from ZIM:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli extract-zim-text \
  --input-zim /data/iiab/zim/wikipedia_en_all_maxi.zim \
  --output data/extracted/wiki_corpus.txt \
  --max-articles 50000 \
  --min-chars 200
```
If extraction returns `written_articles=0`, retry with a lower `--min-chars` (for example `20`).
If `extract-zim-text` reports no fulltext index, generate a `--paths-file` from
ZIM suggestions/title index and rerun extraction with that file.

2. Analyze extracted corpora and generate boilerplate candidates:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli corpus-quality-report \
  --input-dir data/extracted \
  --output artifacts/reports/corpus_quality.json
```

3. Clean corpora before tokenizer training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli clean-corpus-batch \
  --input-dir data/extracted \
  --output-dir data/cleaned \
  --boilerplate-report artifacts/reports/corpus_quality.json
```
By default this cleanup step also decodes HTML entities and strips common web-shell artifacts
(HTML-like tags, repeated nav/menu phrases, site suffixes such as `- Stack Overflow`).
Disable individual transforms with:
`--no-decode-html-entities`, `--no-strip-html-tags`, `--no-strip-site-suffixes`,
`--no-strip-nav-phrases`, `--no-strip-stack-metadata`, `--no-collapse-repeated-prefix`,
`--no-strip-inline-score-tokens`.
To enforce English-only cleanup, add `--en-only` (with tunable thresholds:
`--en-min-words`, `--en-min-stopword-ratio`, `--en-min-stopword-count`,
`--en-min-latin-ratio`).
For talking-only passes, keep code filtering enabled (default) or tune with:
`--code-symbol-ratio-threshold` and `--code-keyword-hits-threshold`.

4. Train tokenizer on cleaned corpus:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer \
  --input data/cleaned/wiki_corpus.clean.txt \
  --output artifacts/tokenizer/vocab.json
```

5. Shard tokenized corpus for training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus \
  --input data/cleaned/wiki_corpus.clean.txt \
  --tokenizer artifacts/tokenizer/vocab.json \
  --output-dir data/shards/wiki_char \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```

5b. Build one global tokenizer for multi-dataset training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer-global \
  --input-dir data/cleaned \
  --pattern "*.clean.txt" \
  --from-shards-path data/shards \
  --output artifacts/tokenizer/global-char-v1.json
```

5c. Re-shard many corpora with that global tokenizer:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus-batch \
  --input-dir data/cleaned \
  --pattern "*.clean.txt" \
  --from-shards-path data/shards \
  --tokenizer artifacts/tokenizer/global-char-v1.json \
  --output-root data/shards_global/global-char-v1
```

6. Inspect corpus quickly:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli stats --input data/cleaned/wiki_corpus.clean.txt
```

7. Verify shard integrity before training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards \
  --raw-zim-dir data/raw_zim \
  --strict-source
```

8. Run a baseline training test:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards/medlineplus.gov_en_all_2025-01 \
  --output-dir artifacts/checkpoints/medlineplus_baseline \
  --max-steps 200 \
  --batch-size 8 \
  --context-length 256
```
Note: `train` requires all selected manifests to share the exact same tokenizer mapping.
Use a global tokenizer + `shard-corpus-batch` output root for multi-dataset runs.

9. Generate text from a checkpoint:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli generate \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/last.pt \
  --prompt "The future of medicine is" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 50
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
- Batch corpus quality report generation (`corpus-quality-report`).
- Batch corpus cleanup and dedupe (`clean-corpus-batch`).
- Basic character-level tokenizer with train/save/load.
- Token-window data pipeline (`TokenWindowDataset`) for next-token training pairs.
- ZIM archive text extraction (`extract-zim-text`) for server-hosted `.zim` files.
  - Automatically falls back to suggestion-index paths if fulltext search returns no matches.
- Corpus sharding (`shard-corpus`) into train/val token shard binaries + manifest.
- Batch corpus sharding (`shard-corpus-batch`) with one shared tokenizer.
- Baseline GPT training (`train`) with checkpoint save/resume.
- Checkpoint-based text generation (`generate`) with temperature/top-k sampling.
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
