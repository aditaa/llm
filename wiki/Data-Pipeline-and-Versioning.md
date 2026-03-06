# Data Pipeline and Versioning

## Pipeline Stages
1. Extract text from `.zim` archives.
2. Train tokenizer vocabulary.
3. Shard tokenized corpus to train/val binary files.
4. Train model from shard manifests.

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli extract-zim-text --input-zim /path/file.zim --output /path/corpus.txt
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer --input /path/corpus.txt --output /path/vocab.json
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus --input /path/corpus.txt --tokenizer /path/vocab.json --output-dir /path/shards
PYTHONPATH=src .venv/bin/python -m llm.cli train --shards-path /path/shards --output-dir /path/checkpoints
```

Heuristic risk audit before tokenizer training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli dataset-risk-report \
  --input-dir data/cleaned \
  --output artifacts/reports/dataset_risk.json
```

Shared tokenizer workflow for multi-dataset training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer-global --input-dir data/extracted --from-shards-path data/shards --output artifacts/tokenizer/global-char-v1.json
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus-batch --input-dir data/extracted --from-shards-path data/shards --tokenizer artifacts/tokenizer/global-char-v1.json --output-root data/shards_global/global-char-v1
PYTHONPATH=src .venv/bin/python -m llm.cli train --shards-path data/shards_global/global-char-v1 --output-dir artifacts/checkpoints/global-char-v1
```

## Versioning Rule
Use ZIM date stamps as the canonical dataset version.

Example:
- ZIM: `serverfault.com_en_all_2025-08.zim`
- Version tag: `serverfault_2025-08`
- Extracted text: `serverfault_2025-08.txt`
- Tokenizer: `serverfault_2025-08-vocab.json`
- Shard folder: `serverfault_2025-08/`

This creates a 1:1 mapping between source snapshot and derived artifacts.

## Storage Layout
Use hot + warm storage:
- Hot working set (default processing location):
  - `data/raw_zim/`
  - `data/extracted/`
  - `data/shards/`
  - `artifacts/tokenizer/`
- Warm cache/backup mount:
- `/mnt/ceph/llm/data/raw_zim/`
- `/mnt/ceph/llm/data/extracted/`
- `/mnt/ceph/llm/data/shards/`
- `/mnt/ceph/llm/data/tokenizer/`

Push local artifacts to warm storage:
```bash
bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data
```

Rehydrate local hot workspace from warm storage:
```bash
bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data
```

## First-Pass Talking Profile
For an English prose-first pass, generate include/exclude manifests and optionally move excluded ZIMs to warm storage:

```bash
bash scripts/first_pass_zim_profile.sh
bash scripts/first_pass_zim_profile.sh --move-excluded
```

Generated manifests:
- `artifacts/reports/first_pass_include_targets.txt`
- `artifacts/reports/first_pass_include_zims.txt`
- `artifacts/reports/first_pass_exclude_zims.txt`

## Pre-Training Integrity Gate
Before training, verify shard datasets:

```bash
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards \
  --raw-zim-dir data/raw_zim \
  --strict-source
```

This validates:
- Manifest consistency and token totals
- Shard file sizes and token counts
- Token ID range against tokenizer vocab
- Optional source ZIM health linkage

## Update Strategy
- Keep old shard versions immutable until new version is validated.
- Switch training to new manifest only after smoke validation.
- Delete stale extracted/shards only when space is needed.
- Train on tokenizer-compatible shard sets only (same tokenizer mapping across selected manifests).
- For no-fulltext ZIM files, generate `--paths-file` from suggestion/title index and use it for extraction.
