# Data Pipeline and Versioning

## Pipeline Stages
1. Extract text from `.zim` archives.
2. Train tokenizer vocabulary.
3. Shard tokenized corpus to train/val binary files.
4. Train model from shard manifests.

FineWeb-first fast path (preferred for first build):
1. Download FineWeb parquet shards.
2. Build tokenizer + train/val shards directly from parquet.
3. Train model from shard manifests.

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli extract-zim-text --input-zim /path/file.zim --output /path/corpus.txt
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer --input /path/corpus.txt --output /path/vocab.json --bpe-vocab-size 32000 --bpe-min-frequency 2
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus --input /path/corpus.txt --tokenizer /path/vocab.json --output-dir /path/shards
PYTHONPATH=src .venv/bin/python -m llm.cli train --shards-path /path/shards --output-dir /path/checkpoints --precision auto
```

Heuristic risk audit before tokenizer training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli dataset-risk-report \
  --input-dir data/cleaned \
  --output artifacts/reports/dataset_risk.json
```

Shared tokenizer workflow for multi-dataset training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer-global --input-dir data/extracted --from-shards-path data/shards --output artifacts/tokenizer/global-bpe-v1.json --bpe-vocab-size 32000 --bpe-min-frequency 2
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus-batch --input-dir data/extracted --from-shards-path data/shards --tokenizer artifacts/tokenizer/global-bpe-v1.json --output-root data/shards_global/global-bpe-v1
PYTHONPATH=src .venv/bin/python -m llm.cli train --shards-path data/shards_global/global-bpe-v1 --output-dir artifacts/checkpoints/global-bpe-v1

Throughput tuning notes:
- Prefer `--precision auto` on CUDA.
- Keep eval overhead bounded (`--eval-interval 500+`, `--eval-steps 5-10`).
- If utilization is bursty, test `--compile-model --compile-mode reduce-overhead`.
```

Direct FineWeb parquet to shards:
```bash
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-10BT \
  --output-dir data/shards_global/fineweb-s10bt-global-bpe-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-s10bt-global-bpe-v1.json \
  --bpe-vocab-size 32000 \
  --field text \
  --min-chars 80 \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```

Rolling FineWeb 350BT ingestion on limited hot disk:
```bash
bash scripts/fineweb_stage_shard_loop.sh \
  --stage-max-files 10 \
  --process-max-files 10 \
  --sleep-seconds 120
```
This loop:
- stages bounded parquet files from warm (`/mnt/ceph/llm/data`) to hot (`./data`)
- builds shard batches with a shared tokenizer
- runs `verify-shards` on each batch
- syncs shard outputs back to warm storage
- deletes processed hot parquet files to reclaim local space

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

For FineWeb shard manifests, run without `--raw-zim-dir`.

## Update Strategy
- Keep old shard versions immutable until new version is validated.
- Switch training to new manifest only after smoke validation.
- Delete stale extracted/shards only when space is needed.
- Train on tokenizer-compatible shard sets only (same tokenizer mapping across selected manifests).
- For no-fulltext ZIM files, generate `--paths-file` from suggestion/title index and use it for extraction.
