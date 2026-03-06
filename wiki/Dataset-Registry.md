# Dataset Registry

Track every source we ingest, why it exists, and where it lives.

## Active Sources
| Source | Purpose | Stage | Storage | Status | Notes |
|---|---|---|---|---|---|
| IIAB/Kiwix ZIM collections | Core broad-domain pretraining text | Extraction -> cleanup -> sharding | Hot: `data/raw_zim`, Warm: `/mnt/ceph/llm/data/raw_zim` | Active | Version by ZIM date stamp in filename |
| FineWeb `sample-10BT` parquet | Broad web pretraining text for first pass | Bulk download -> direct parquet-to-shards | Hot: `data/fineweb/sample-10BT` | Active | ~30.6 GB total |
| FineWeb `sample-350BT` parquet | Larger web corpus staged for later scale-up | Bulk download -> warm cache -> staged hot chunks | Warm: `/mnt/ceph/llm/data/fineweb/sample-350BT` | Active (downloading) | ~1.06 TB total |

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

# sample-10BT parquet to hot storage
HF_HUB_DISABLE_XET=1 .venv/bin/hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/*.parquet" \
  --local-dir data/fineweb/sample-10BT \
  --max-workers 2 \
  --token "$HF_TOKEN"

# sample-350BT parquet to warm storage
HF_HUB_DISABLE_XET=1 .venv/bin/hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/350BT/*.parquet" \
  --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
  --max-workers 2 \
  --token "$HF_TOKEN"

# stage bounded chunks from warm -> hot
bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8

# build tokenizer + shards directly from sample-10BT parquet
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-10BT \
  --output-dir data/shards_global/fineweb-s10bt-global-char-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-s10bt-global-char-v1.json \
  --field text

# verify before training
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards_global/fineweb-s10bt-global-char-v1
```
