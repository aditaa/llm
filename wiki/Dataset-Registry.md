# Dataset Registry

Track every source we ingest, why it exists, and where it lives.

## Active Sources
| Source | Purpose | Stage | Storage | Status | Notes |
|---|---|---|---|---|---|
| IIAB/Kiwix ZIM collections | Core broad-domain pretraining text | Extraction -> cleanup -> sharding | Hot: `data/raw_zim`, Warm: `/mnt/ceph/llm/data/raw_zim` | Active | Version by ZIM date stamp in filename |
| `gutenberg_en_all_2025-11.zim` | Long-form English prose for language fluency | Extraction -> cleanup -> sharding | Hot + Warm | Ingesting | Copyright-safe because source is public-domain corpus packaging |
| FineWeb `sample-10BT` rows slice | Add broad web language diversity with bounded size | Pull rows -> cleanup -> sharding | Warm first: `/mnt/ceph/llm/data/extracted/` | Planned/starting | Use bounded rows via `scripts/pull_hf_rows.py` |

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
# Bounded FineWeb pull to warm storage
python3 scripts/pull_hf_rows.py \
  --dataset HuggingFaceFW/fineweb \
  --config sample-10BT \
  --split train \
  --output /mnt/ceph/llm/data/extracted/fineweb_sample-10BT_rows100k.txt \
  --max-rows 100000

# Heuristic risk audit on the pulled file once staged in a directory
PYTHONPATH=src .venv/bin/python -m llm.cli dataset-risk-report \
  --input-dir /mnt/ceph/llm/data/extracted \
  --pattern "fineweb_sample-10BT_rows100k.txt" \
  --output artifacts/reports/dataset_risk_fineweb_sample-10BT_rows100k.json
```
