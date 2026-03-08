# Dataset Registry

Track every source we ingest, why it exists, and where it lives.

## Active Sources
| Source | Purpose | Stage | Storage | Status | Notes |
|---|---|---|---|---|---|
| IIAB/Kiwix ZIM collections | Core broad-domain pretraining text | Extraction -> cleanup -> sharding | Hot: `data/raw_zim`, Warm: `/mnt/ceph/llm/data/raw_zim` | Active | Version by ZIM date stamp in filename |
| FineWeb `sample-350BT` parquet | Primary first-pass web pretraining corpus | Bulk download -> warm cache -> staged hot chunks -> sharding | Warm: `/mnt/ceph/llm/data/fineweb/sample-350BT`, Hot stage: `data/fineweb/sample-350BT` | Active (downloading) | ~1.06 TB total |

## Deprecated Sources
| Source | Reason | Action |
|---|---|---|
| FineWeb `sample-10BT` parquet | Fully contained in `sample-350BT` | Do not ingest new 10BT files; remove hot copies when space is needed |

## Candidate Sources (Not Yet Approved)
| Source | Potential Use | Main Risk to Review |
|---|---|---|
| FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) | Higher educational density | Classifier-filtered curation decisions |
| Dolma | Broad coverage and scale | Mixed-source behavior drift and license mix |
| Dialogue corpora (OpenSubtitles, OASST1, DailyDialog) | Conversational tuning/eval | License constraints, refusal-style contamination, higher toxicity variance |

## Approval Rules
1. Record source URL, license, snapshot date, and checksum/manifest details.
2. Run `dataset-risk-report` on pulled text before tokenizer training.
3. Keep raw or pulled source copies in warm storage before deletion from hot storage.
4. Mark each source as `Active`, `Planned`, `Paused`, or `Rejected`.

## Commands
```bash
# HF token (recommended): Hugging Face web UI -> Settings -> Access Tokens (read)
export HF_TOKEN=hf_xxx

# sample-350BT parquet to warm storage (resumable)
bash scripts/hf_download_resumable.sh \
  --dataset HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/350BT/*.parquet" \
  --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
  --max-workers 4 \
  --enable-hf-transfer \
  --skip-dry-run \
  --attempt-timeout-seconds 5400 \
  --retry-delay-seconds 30 \
  --max-retries 0 \
  --log-file artifacts/reports/fineweb_350bt_download_resumable.log

# stage bounded chunks from warm -> hot
bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8

# build tokenizer + shards directly from staged sample-350BT parquet
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --output-dir data/shards_global/fineweb-global-bpe-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json \
  --bpe-vocab-size 32000 \
  --bpe-min-frequency 2 \
  --field text

# verify before training
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards_global/fineweb-global-bpe-v1
```
