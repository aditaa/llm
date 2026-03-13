# Data Pipeline and Versioning

## Pipeline Stages
1. Extract text from `.zim` archives.
2. Clean/filter corpus (`clean-corpus-batch`) with English + noise gates.
3. Train tokenizer vocabulary.
4. Shard tokenized corpus to train/val binary files.
5. Train model from shard manifests.

FineWeb-first fast path (preferred for first build):
1. Download FineWeb parquet shards.
2. Build tokenizer + train/val shards directly from parquet.
3. Train model from shard manifests.

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli extract-zim-text --input-zim /path/file.zim --output /path/corpus.txt
PYTHONPATH=src .venv/bin/python -m llm.cli clean-corpus-batch --input-dir /path/extracted --output-dir /path/cleaned --en-only
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer --input /path/cleaned/corpus.clean.txt --output /path/vocab.json --bpe-vocab-size 32000 --bpe-min-frequency 2
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus --input /path/cleaned/corpus.clean.txt --tokenizer /path/vocab.json --output-dir /path/shards
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
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer-global --input-dir data/cleaned --pattern "*.clean.txt" --from-shards-path data/shards --output artifacts/tokenizer/global-bpe-v1.json --bpe-vocab-size 32000 --bpe-min-frequency 2
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus-batch --input-dir data/cleaned --pattern "*.clean.txt" --from-shards-path data/shards --tokenizer artifacts/tokenizer/global-bpe-v1.json --output-root data/shards_global/global-bpe-v1
PYTHONPATH=src .venv/bin/python -m llm.cli train --shards-path data/shards_global/global-bpe-v1 --output-dir artifacts/checkpoints/global-bpe-v1
```

Cleaning defaults now include:
- normalized dedupe across punctuation/case variants (disable with `--no-dedupe-normalized`)
- contamination filtering for benchmark/prompt/refusal fragments (disable with `--no-drop-contamination`)
- extendable contamination regexes via `--contamination-pattern` or `--contamination-patterns-file`

Throughput tuning notes:
- Prefer `--precision auto` on CUDA.
- Keep eval overhead bounded (`--eval-interval 500+`, `--eval-steps 5-10`).
- Prefer warmup+cosine LR (`--lr-schedule cosine --lr-warmup-steps <N>`).
- Use `--grad-accum-steps` to trade throughput for lower peak VRAM.
- Keep held-out eval batches frozen (default) and optionally gate regressions with
  `--fail-on-eval-regression --eval-regression-tolerance 0.20`.
- Enable EMA for long runs with `--ema-decay 0.999 --ema-start-step <warmup_end>`.
- If utilization is bursty, test `--compile-model --compile-mode reduce-overhead`.

Direct FineWeb parquet to shards:
```bash
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --output-dir data/shards_global/fineweb-global-bpe-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json \
  --bpe-vocab-size 32000 \
  --field text \
  --min-chars 80 \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```

Rolling FineWeb 350BT ingestion on limited hot disk:
```bash
bash scripts/fineweb_stage_shard_loop.sh \
  --hot-queue-min-files 18 \
  --stage-max-files 12 \
  --stage-copy-jobs 2 \
  --stage-min-free-gib 80 \
  --process-max-files 12 \
  --shard-jobs 2 \
  --auto-tune-shard-jobs \
  --auto-tune-min-shard-jobs 1 \
  --auto-tune-max-shard-jobs 4 \
  --tokenizer-threads 10 \
  --encode-batch-size 1024 \
  --shard-size-tokens 20000000 \
  --sync-background \
  --sync-max-inflight 2 \
  --sleep-seconds 60 \
  --shard-min-batch-size 512
```
This loop:
- stages bounded parquet files from warm (`/mnt/ceph/llm/data`) to hot (`./data`)
- builds shard batches with a shared tokenizer
- runs `verify-shards` on each batch
- syncs shard outputs back to warm storage
- deletes processed hot parquet files to reclaim local space
- retries shard builds automatically on OOM-like failures by reducing batch size
- preflights selected parquet files (row groups/rows/required `text` field)
- quarantines bad hot parquet files and tracks them in `artifacts/reports/fineweb_stage_shard_loop/bad_parquet_files.txt`
- marks files processed only after post-batch guardrails pass (valid report+manifest, non-empty shard files)
- enforces a hot-disk free-space floor when `--stage-min-free-gib` is set
- can auto-tune shard parallelism (`--auto-tune-shard-jobs`) using CPU load + per-batch runtime
- can overlap warm sync in the background (`--sync-background`) with bounded in-flight jobs
- benefits from larger shard targets (`--shard-size-tokens 20000000`) to cut file-count/sync overhead
- quarantines shard-job inputs on non-OOM shard-build failures and continues with remaining files
- on 20-core hosts, two shard jobs with tokenizer batch encoding is the current higher-throughput profile
- reconciles bad parquet basenames against warm-source validity on startup, so transient hot-copy failures can be retried

Optional watchdog for stage/shard loop auto-restart:
```bash
bash scripts/fineweb_stage_shard_watchdog.sh \
  --worker-args "--hot-queue-min-files 10 --stage-max-files 8 --stage-copy-jobs 4 --stage-min-free-gib 80 --process-max-files 15 --shard-jobs 2 --auto-tune-shard-jobs --auto-tune-min-shard-jobs 2 --auto-tune-max-shard-jobs 3 --auto-tune-low-load-pct 80 --auto-tune-high-load-pct 95 --auto-tune-min-batch-seconds 300 --tokenizer-threads 10 --encode-batch-size 1024 --shard-size-tokens 20000000 --sync-background --sync-max-inflight 2 --sleep-seconds 60 --shard-min-batch-size 512" \
  --check-interval-seconds 120 \
  --stall-seconds 5400
```
The stage watchdog enforces a singleton lock, stops worker process groups cleanly during restarts,
and now cleans stale stage-loop/shard-worker processes before relaunching.

Optional watchdog for large FineWeb downloads:
```bash
bash scripts/hf_download_watchdog.sh \
  --dataset HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/350BT/*.parquet" \
  --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
  --max-workers 4 \
  --enable-hf-transfer \
  --skip-dry-run \
  --attempt-timeout-seconds 5400 \
  --stall-seconds 1200 \
  --exit-on-complete \
  --expected-parquet-files 510 \
  --expected-bytes 1061360917731
```
The watchdog monitors parquet/incomplete byte growth and restarts the resumable worker if it exits or stalls.
Use `--exit-on-complete` with expected targets so the watchdog exits when the download is actually complete.

Simple full FineWeb-Edu sync (direct to Ceph path):
```bash
export HF_TOKEN=hf_xxx   # optional but recommended
bash scripts/sync_fineweb_edu_full.sh /mnt/pve/cephfs/llm/data/fineweb/fineweb-edu-full
```

Hot-queue prefetch worker (stage on demand while training):
```bash
bash scripts/fineweb_prefetch_hot_queue.sh \
  --queue-min-files 12 \
  --stage-max-files 8 \
  --sleep-seconds 60
```
`scripts/stage_fineweb_from_warm.sh` now stages parquet files via
`*.parquet.incomplete` temp files and atomically renames on completion, so
downstream preflight/sharding does not see partially written parquet data.
Use `--copy-jobs <N>` on stage commands to parallelize warm->hot copies.
Use `--min-free-gib <N>` on stage commands to stop staging before hot disk gets too full.

Auto-resume trainer supervisor for growing shard sets:
```bash
bash scripts/train_supervisor_rtx5070_350bt.sh \
  --step-chunk 2000 \
  --poll-seconds 60 \
  --batch-size 12 \
  --target-effective-batch 24 \
  --min-unique-input-files 510 \
  --min-batch-size 6 \
  --max-batch-size 20 \
  --batch-step 2 \
  --generation-suite configs/eval/generation_smoke_suite_v1.json \
  --generation-every-chunks 1
```
Phase-1 English conversation gate profile:
```bash
bash scripts/train_supervisor_phase1_english_talk.sh
```
This profile uses `configs/eval/english_talk_suite_v1.json`,
`configs/eval/generation_talk_smoke_v1.json`, and
`configs/eval/promotion_policy_talk_v1.json` before code-specialization passes.

This runs training in chunks and resumes from `last.pt`; each chunk restart re-reads
all manifests under `data/shards_global/fineweb-global-bpe-v1` so newly added shard
batches are picked up without manual intervention.
Use `--min-unique-input-files <N>` to avoid starting training before enough parquet
coverage is represented in shard manifests.
Use `--min-train-tokens <N>` when you want readiness gated by total train-token coverage.
Use `--dedupe-report-keep <N>` to cap saved dedupe report/log artifacts during long waits.
Resume guardrail now validates `last.pt`/`ckpt_step_*.pt` for resumability and
quarantines invalid files before fallback to the newest valid checkpoint.
Use `--no-train-fail-on-eval-regression` if you want chunk training to continue and
primarily gate via the post-chunk prompt-suite regression/promotion checks.
Trend outputs:
- `artifacts/reports/train_supervisor_350bt/train_trend.tsv`
- `artifacts/reports/train_supervisor_350bt/eval_trend.tsv`
- `artifacts/reports/train_supervisor_350bt/generation_trend.tsv`
- `artifacts/reports/train_supervisor_350bt/eval_dashboard.html`
The supervisor eval step now auto-selects the latest successful baseline from the
same suite name/path (same behavior for generation-gate baselines), compares deltas
(pass/check/score), and applies promotion policy checks when configured.
Supervisor now auto-promotes `best.pt` aliases after successful eval promotion checks.

For long runs with bounded disk use, pass checkpoint retention options:
`--checkpoint-keep-last 6 --checkpoint-keep-every 10000`

For context-extension stage from a 512-token checkpoint:
`--resume-from <last.pt> --context-length 1024 --allow-context-extension`

Checkpoint smoothing after training/eval:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli average-checkpoints \
  --checkpoint artifacts/checkpoints/fineweb-350bt-run1/ckpt_step_0002000.pt \
  --checkpoint artifacts/checkpoints/fineweb-350bt-run1/ckpt_step_0003000.pt \
  --output artifacts/checkpoints/fineweb-350bt-run1/avg_last2.pt \
  --state-key model_state
```

Combined ETA/status reporter:
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_eta_report.py --loop --interval-seconds 60
```
Outputs:
- `artifacts/reports/pipeline_status.json`
- `artifacts/reports/pipeline_status.txt`
Includes system snapshots from `top -b -n1`, `free -h`, `nvidia-smi`, and `df -h`.

Live terminal view:
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_live_view.py --refresh-seconds 5
```
This viewer is live-only (no report file writes) and refreshes in-place with:
- system status (CPU, memory, GPU, disk)
- pipeline progress (download/staging/sharding/training)
- running task status (pid/runtime/cpu/mem summary)
Add `--no-alt-screen` if your terminal does not render full-screen updates correctly.

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

Pre-wipe safety checklist:
```bash
git fetch --all --prune
git status --short --branch
git push
bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data
stamp=$(date -u +%Y%m%dT%H%M%SZ)
git ls-files --others --ignored --exclude-standard | sort \
  > /mnt/ceph/llm/data/logs/git_untracked_all_${stamp}.txt
```
Only wipe after sync completion and a fresh `last_sync_utc.txt` timestamp.

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
- Manifest compatibility checks include `tokenizer_hash` and `tokenizer_contract_hash`.
- For no-fulltext ZIM files, generate `--paths-file` from suggestion/title index and use it for extraction.
