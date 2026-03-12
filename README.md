# LLM From Scratch

[![CI](https://github.com/aditaa/llm/actions/workflows/ci.yml/badge.svg)](https://github.com/aditaa/llm/actions/workflows/ci.yml)
[![Wiki Sync](https://github.com/aditaa/llm/actions/workflows/wiki-sync.yml/badge.svg)](https://github.com/aditaa/llm/actions/workflows/wiki-sync.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#)

Production-style, decoder-only LLM engineering project focused on reproducible data pipelines, tokenizer/sharding workflows, and GPU training from scratch.

## About
- Scope: end-to-end LLM workflow from raw corpora to checkpoints and generation.
- Data focus: ZIM + FineWeb workflows with hot (`./data`) and warm (`/mnt/ceph/llm/data`) storage patterns.
- Engineering focus: deterministic scripts, integrity checks, CI gating, and wiki-backed docs.

## Quick Links
- Wiki: [`wiki/`](wiki)
- Setup: [`docs/SERVER_SETUP.md`](docs/SERVER_SETUP.md)
- RTX 5070 tuning: [`docs/RTX5070_TUNING.md`](docs/RTX5070_TUNING.md)
- HF release + deploy: [`docs/HF_RELEASE_AND_DEPLOY.md`](docs/HF_RELEASE_AND_DEPLOY.md)
- Contributor guide: [`AGENTS.md`](AGENTS.md)

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
make setup-infer # install inference/deploy dependencies
make install-systemd-services # install/reload long-run systemd units
make install-user-systemd-services # install/reload user-level systemd units (no sudo; includes shard/checkpoint timers)
make test        # run unit tests
make lint        # run Ruff checks
make format      # run Black formatter
make typecheck   # run MyPy
make smoke       # tiny CLI smoke check
make verify-shards # print shard integrity check usage
make train       # print baseline training command usage
make generate    # print checkpoint text-generation command usage
make average-checkpoints # print checkpoint averaging usage
make eval-checkpoint # print standardized prompt-suite eval usage
make render-eval-dashboard # print eval trend dashboard render usage
make package-inference-bundle # print deploy bundle packaging usage
make train-tokenizer-global # print shared-tokenizer command usage
make corpus-quality-report # print quality report command usage
make clean-corpus-batch # print batch cleanup command usage
make dataset-risk-report # print heuristic dataset risk audit command usage
make pull-hf-rows # print Hugging Face rows API pull helper usage
make fineweb-parquet-to-shards # print direct FineWeb parquet->token-shards usage
make fineweb-manifest-dedupe # print overlap-manifest dedupe helper usage
make stage-fineweb-from-warm # print warm->hot FineWeb chunk staging usage
make fineweb-revalidate-bad-parquet # print bad parquet revalidate/restage usage
make reconcile-offloaded-manifests # print offloaded-manifest reconcile usage
make shard-offload-cycle # print safe reconcile->offload->reconcile usage
make offload-shard-bins-warm # print shard .bin offload-to-warm usage
make hot-shard-warmup # print active-shard warm->hot hydration usage
make fineweb-stage-shard-loop # print rolling stage->shard->verify->sync->purge usage
make fineweb-stage-shard-watchdog # print auto-restart watchdog usage for stage/shard loop
make lr-sweep-350bt # print RTX 5070 LR sweep usage for staged 350BT shards
make benchmark-rtx5070 # print short context/batch throughput benchmark usage
make train-350bt-v2 # print 350BT long-run launcher usage
make train-350bt-ctx1024 # print long-context continuation launcher usage
make train-supervisor-350bt # print auto-resume trainer supervisor usage
make train-supervisor-phase1-talk # print phase-1 English conversation supervisor usage
make pipeline-eta # print combined download/shard/train ETA reporter usage
make pipeline-live # print live terminal pipeline dashboard usage
make shard-corpus-batch # print shared-tokenizer batch sharding usage
make hf-download-resumable # print self-healing HF resume-download worker usage
make hf-download-watchdog # print auto-restart wrapper for stalled/exited HF downloads
make sync-warm   # sync raw/training data + artifacts to warm storage
make hydrate-warm # hydrate hot workspace from warm storage
make offload-zim # continuously move raw ZIMs hot -> warm
make checkpoint-offload-prune # sync checkpoints to warm and prune older local runs
make checkpoint-step-offload # offload older ckpt_step_*.pt while keeping newest local resume steps
make set-swappiness # print vm.swappiness tuning usage (root)
make hf-prepare-publish # print HF bundle/publish usage
make hf-download-model # print full HF model download usage
make serve-openai # print local OpenAI-compatible server usage
make doctor      # verify binaries and Python deps
```
`make smoke` is expected to run in CI without installing `torch`; keep non-training CLI import paths torch-optional.

## CI/CD
GitHub Actions workflows are defined in `.github/workflows/`:
- `ci.yml`: script sanity (`bash -n` + `py_compile`), lint, typecheck, unit tests, smoke checks on pull requests and pushes to `main`
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
5. Install persistent workers:
   - system units (root): `bash scripts/install_systemd_services.sh --install-watchdog`
   - user units (no sudo): `bash scripts/install_user_systemd_services.sh --install-watchdog`

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
  --boilerplate-report artifacts/reports/corpus_quality.json \
  --en-only
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
Additional quality guards are enabled by default:
- minimum words per line (`--min-words`, default `6`)
- symbol-density filter (`--max-symbol-ratio`, default `0.20`)
- URL-heavy line filter (`--max-urls-per-line`, default `1`)
- repetitive-token noise filter (`--repeated-token-run-threshold`, default `8`)
- normalized dedupe keys across punctuation/case variants (`--no-dedupe-normalized` to disable)
- contamination filter for benchmark/prompt/refusal fragments (`--no-drop-contamination` to disable)
For talking-only passes, keep code filtering enabled (default) or tune with:
`--code-symbol-ratio-threshold` and `--code-keyword-hits-threshold`.
You can extend contamination filtering with repeatable `--contamination-pattern` or
`--contamination-patterns-file`.

3a. Pull a bounded Hugging Face dataset slice (for example FineWeb sample rows):
```bash
python3 scripts/pull_hf_rows.py \
  --dataset HuggingFaceFW/fineweb \
  --config sample-350BT \
  --split train \
  --output /mnt/ceph/llm/data/extracted/fineweb_sample-350BT_rows100k.txt \
  --max-rows 100000
```
Use warm storage for these pulls first; full FineWeb variants are much larger than typical hot disk.

3aa. Bulk-download FineWeb parquet shards (resumable):
```bash
# create token in Hugging Face web UI: Settings -> Access Tokens (read scope)
export HF_TOKEN=hf_xxx

# sample-350BT (~1.06 TB) -> warm storage, auto-resume + retry forever
bash scripts/hf_download_resumable.sh \
  --dataset HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/350BT/*.parquet" \
  --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT \
  --max-workers 6 \
  --enable-hf-transfer \
  --skip-dry-run \
  --attempt-timeout-seconds 5400 \
  --retry-delay-seconds 30 \
  --max-retries 0 \
  --log-file artifacts/reports/fineweb_350bt_download_resumable.log
```
Notes:
- `HF_TOKEN` is recommended (higher limits), not strictly required for public datasets.
- Hugging Face SSH keys are for Git-over-SSH and are not used by `hf download`.
- `hf_download_resumable.sh` writes a lock file in the local dir to prevent duplicate workers.
- `hf_download_resumable.sh` auto-detects `hf_transfer` and can be forced with `--enable-hf-transfer`.
- For very large pulls (like 350BT), `--skip-dry-run` avoids metadata preflight stalls.
- `--attempt-timeout-seconds` prevents one hung transfer from stalling progress forever.
- Keep 350BT parquet on warm storage and stage bounded chunks to hot storage before sharding.
- For unattended runs, wrap with `scripts/hf_download_watchdog.sh` to auto-restart on stalls.

3aaa. Optional watchdog wrapper for stalled/exited downloads:
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
  --expected-bytes 1061360917731 \
  --worker-log-file artifacts/reports/fineweb_350bt_download_resumable.log \
  --watchdog-log-file artifacts/reports/hf_download_watchdog.log
```
Use `--exit-on-complete` with expected file and/or byte targets so the watchdog exits once
download is complete (instead of looping forever and relaunching workers).

3ab. (Optional legacy) Stage FineWeb chunks from warm to hot:
```bash
bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8 --copy-jobs 2
```
You can pass `--skip-list artifacts/reports/fineweb_stage_shard_loop/bad_parquet_files.txt`
to avoid restaging files previously flagged as invalid.
Use `--min-free-gib <N>` to keep a floor of free space on hot storage while staging.
The staging script now copies into `*.parquet.incomplete` first and renames atomically,
so sharding/preflight never reads partially written parquet files.

3ac. Run direct-from-Ceph sharding loop (recommended baseline):
```bash
bash scripts/fineweb_stage_shard_loop.sh \
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
  --hot-max-used-pct 80 \
  --sleep-seconds 60 \
  --shard-min-batch-size 512
```
This loop reads parquet directly from Ceph (`/mnt/ceph/llm/data/...` by default),
builds verified shard batches under `data/shards_global/fineweb-global-bpe-v1/`,
and syncs each successful batch back to warm storage immediately.
It keeps sharding moving while automatically offloading older already-trained shard batches
when hot-disk usage exceeds `--hot-max-used-pct` (default 80), rather than pausing sharding.
Use `--enable-stage-copy` only if you explicitly want warm->hot parquet staging.
Before sharding each batch, the loop now runs a parquet preflight check (row groups/rows/field),
quarantines failing hot files, and records their basenames in
`artifacts/reports/fineweb_stage_shard_loop/bad_parquet_files.txt` so they are skipped in future staging.
It also bootstraps processed parquet basenames from existing shard manifests on startup,
builds a combined stage skip list (`processed + bad`), and removes already-known files from hot storage,
so restarted loops continue forward instead of re-staging the earliest parquet files.
It also reconciles `bad_parquet_files.txt` against warm-source parquet validity on startup, so
transient hot-copy failures do not permanently blacklist valid warm files.
In direct-source mode, `--hot-queue-min-files`/`--stage-*` options are inactive unless
`--enable-stage-copy` is set.
`--auto-tune-shard-jobs` adapts `--shard-jobs` (and matching tokenizer threads) from loadavg + batch runtime.
`--sync-background` overlaps warm-storage sync with the next shard batch to reduce idle gaps.
`--shard-size-tokens 20000000` reduces shard file-count overhead vs the old 5M-token default.
If a shard build fails with OOM-like errors, the loop retries automatically with a smaller batch size.
Batch guardrails now require valid report/manifest + non-empty shard outputs before files are marked
processed or purged from hot storage.
If a shard build fails with non-OOM corruption errors (for example parquet decode errors),
the loop can attempt a one-time warm->hot restage retry in stage-copy mode.
If retry still fails, inputs are quarantined as bad and the loop continues with remaining files.
Additionally, newly staged parquet files are deep-validated before shard build
(configurable via `--deep-validate-max-batches` and `--deep-validate-batch-size`)
to catch decode corruption earlier and quarantine files sooner.
Use `--parquet-validate-timeout-seconds` to cap validation calls and avoid hangs on a single Ceph file.
Guardrail checks are implemented in `src/llm/fineweb_guardrails.py` and are unit-tested.
For 20-core hosts, `--shard-jobs 2 --tokenizer-threads 10 --encode-batch-size 1024` is the
current high-throughput profile.
After full coverage (`--expected-unique-input-files`, default `510`), the loop now
switches into training-focused mode and pauses additional stage/shard churn.

3ad. Optional watchdog for stage/shard loop auto-restart on exit/stall:
```bash
bash scripts/fineweb_stage_shard_watchdog.sh \
  --worker-args "--process-max-files 12 --shard-jobs 2 --tokenizer-threads 10 --encode-batch-size 1024 --hot-max-used-pct 80 --sleep-seconds 60 --shard-min-batch-size 512" \
  --check-interval-seconds 120 \
  --stall-seconds 5400
```
The stage-watchdog now enforces a singleton lock in the stage state directory
(`artifacts/reports/fineweb_stage_shard_loop/watchdog.lock` by default), independent of
the log filename. It also adopts an already-running stage-loop controller by default so
watchdog restarts do not leave direct loop runs unmanaged. Use `--no-adopt-existing-loop`
to force launching a fresh worker process.
Watchdog progress snapshots now include hot `.incomplete` file count/bytes, so long warm->hot
copy phases are treated as active progress (not false stalls).
Watchdog now also suppresses stall restarts after coverage completion
(`--expected-unique-input-files`).

3ad. Build tokenizer + token shards directly from FineWeb parquet:
```bash
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --output-dir data/shards_global/fineweb-global-bpe-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json \
  --field text \
  --min-chars 80 \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```
This writes `manifest.json` + shard `.bin` files directly, skipping extracted text.
Use `--max-files` to do bounded test runs.

3b. Run heuristic dataset risk audit:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli dataset-risk-report \
  --input-dir data/cleaned \
  --output artifacts/reports/dataset_risk.json
```
This reports lexical cues for toxicity, stereotypes, political content, and refusal-like phrases.
Use it as a screening signal, then manually review high-risk segments.

4. Train tokenizer on cleaned corpus:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer \
  --input data/cleaned/wiki_corpus.clean.txt \
  --output artifacts/tokenizer/vocab.json \
  --bpe-vocab-size 32000 \
  --bpe-min-frequency 2
```

5. Shard tokenized corpus for training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus \
  --input data/cleaned/wiki_corpus.clean.txt \
  --tokenizer artifacts/tokenizer/vocab.json \
  --output-dir data/shards/wiki_bpe \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01
```

5b. Build one global tokenizer for multi-dataset training:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train-tokenizer-global \
  --input-dir data/cleaned \
  --pattern "*.clean.txt" \
  --from-shards-path data/shards \
  --output artifacts/tokenizer/global-bpe-v1.json \
  --bpe-vocab-size 32000 \
  --bpe-min-frequency 2
```

5c. Re-shard many corpora with that global tokenizer:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli shard-corpus-batch \
  --input-dir data/cleaned \
  --pattern "*.clean.txt" \
  --from-shards-path data/shards \
  --tokenizer artifacts/tokenizer/global-bpe-v1.json \
  --output-root data/shards_global/global-bpe-v1
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
  --context-length 256 \
  --lr-schedule cosine \
  --lr-warmup-steps 50 \
  --grad-accum-steps 1 \
  --fail-on-eval-regression \
  --precision auto
```
Note: `train` requires all selected manifests to share the exact same tokenizer mapping.
Use a global tokenizer + `shard-corpus-batch` output root for multi-dataset runs.
For higher sustained GPU utilization on CUDA, use `--precision auto` and keep
validation less frequent (`--eval-interval 500 --eval-steps 10`).
If utilization is still bursty on smaller models, test `--compile-model`.
`--compile-model` now warms the compiled graph and safely falls back to eager mode on
backend/warmup failures; use `--compile-strict` to hard-fail instead.
Training now supports:
- warmup + cosine LR schedule (`--lr-schedule`, `--lr-warmup-steps`, `--lr-min-ratio`)
- gradient accumulation (`--grad-accum-steps`)
- fixed held-out eval batches (`--no-eval-freeze-batches` to disable)
- eval regression gate (`--fail-on-eval-regression --eval-regression-tolerance 0.20`)
- checkpoint retention pruning (`--checkpoint-keep-last`, `--checkpoint-keep-every`)
- optional EMA weights (`--ema-decay`, `--ema-update-every`, `--ema-start-step`)
- optional weights-only export (`--export-safetensors`)
- shard sampler FD guardrail (`--sampler-max-open-shards`) to avoid too-many-open-files
- balanced shuffled shard cycling (`--sampler-strategy balanced`) for even shard usage
- minimum full-pass gate (`--sampler-min-full-passes <X>`) to enforce at-least-X passes
- sampled shard trace export (`--sampled-shards-trace`) for per-chunk true coverage accounting

9. Generate text from a checkpoint:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli generate \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/last.pt \
  --prompt "The future of medicine is" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 50
```
Use `--use-ema` to generate from `ema_state` when the checkpoint includes EMA weights.

9a. Average multiple checkpoints for a more stable inference snapshot:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli average-checkpoints \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/ckpt_step_0001000.pt \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/ckpt_step_0002000.pt \
  --output artifacts/checkpoints/medlineplus_baseline/avg_last2.pt \
  --state-key model_state
```

10. Run standardized checkpoint eval (fixed prompt suite + scored report):
```bash
PYTHONPATH=src .venv/bin/python scripts/eval_checkpoint_prompts.py \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/last.pt \
  --suite configs/eval/standard_prompt_suite_v3.json \
  --baseline-report artifacts/reports/evals/<previous_report>.json \
  --promotion-policy configs/eval/promotion_policy_v1.json \
  --fail-on-regression
```
Writes a JSON report under `artifacts/reports/evals/` so runs can be compared over time.
The report now includes `regression` deltas and a `promotion` verdict when a policy is provided.

## FineWeb-Only First-Pass Training
Use this when you want round-1 pretraining only from FineWeb (no ZIM mix yet):

```bash
# 1) build tokenizer + shards directly from parquet
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --output-dir data/shards_global/fineweb-global-bpe-v1 \
  --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json \
  --field text \
  --min-chars 80 \
  --shard-size-tokens 5000000 \
  --val-ratio 0.01

# 2) verify and train
PYTHONPATH=src .venv/bin/python -m llm.cli verify-shards \
  --path data/shards_global/fineweb-global-bpe-v1

PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/fineweb-350bt-run1 \
  --device cuda \
  --max-steps 1000 \
  --batch-size 12 \
  --context-length 256 \
  --lr-schedule cosine \
  --lr-warmup-steps 200 \
  --fail-on-eval-regression \
  --precision auto
```

Resume training from the latest checkpoint:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-global-bpe-v1 \
  --output-dir artifacts/checkpoints/fineweb-350bt-run1 \
  --device cuda \
  --resume-from artifacts/checkpoints/fineweb-350bt-run1/last.pt \
  --max-steps 3000
```

Long-context continuation from a converged ctx512 run:
```bash
bash scripts/train_rtx5070_fineweb_350bt_bpe_v2_ctx1024.sh
```
This path resumes from the base run and uses `--allow-context-extension`.

### Incremental FineWeb Adds While Training
You can start training on a subset, then add new parquet files with the same tokenizer and resume:

```bash
# phase 1 file snapshot (example: first 10 files)
find data/fineweb/sample-350BT/sample/350BT -maxdepth 1 -type f -name '*.parquet' | sort | head -n 10 | sed 's#^data/fineweb/sample-350BT/##' > artifacts/reports/fineweb_sample350bt_phase1_files.txt

# build phase 1 tokenizer + shards
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --files-list artifacts/reports/fineweb_sample350bt_phase1_files.txt \
  --output-dir data/shards_global/fineweb-350bt-incremental/phase1 \
  --tokenizer-out artifacts/tokenizer/fineweb-350bt-incremental-bpe-v1.json \
  --field text

# start training on phase 1
PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-350bt-incremental \
  --output-dir artifacts/checkpoints/fineweb-350bt-incremental-run1 \
  --device cuda

# later: build phase 2 from newly arrived files using same tokenizer
find data/fineweb/sample-350BT/sample/350BT -maxdepth 1 -type f -name '*.parquet' | sort | sed 's#^data/fineweb/sample-350BT/##' > /tmp/all_parquets.txt
comm -23 /tmp/all_parquets.txt artifacts/reports/fineweb_sample350bt_phase1_files.txt > artifacts/reports/fineweb_sample350bt_phase2_files.txt
PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py \
  --input-dir data/fineweb/sample-350BT \
  --files-list artifacts/reports/fineweb_sample350bt_phase2_files.txt \
  --output-dir data/shards_global/fineweb-350bt-incremental/phase2 \
  --tokenizer-in artifacts/tokenizer/fineweb-350bt-incremental-bpe-v1.json \
  --field text

# resume; train sees both manifests under shards-path
PYTHONPATH=src .venv/bin/python -m llm.cli train \
  --shards-path data/shards_global/fineweb-350bt-incremental \
  --output-dir artifacts/checkpoints/fineweb-350bt-incremental-run1 \
  --device cuda \
  --resume-from artifacts/checkpoints/fineweb-350bt-incremental-run1/last.pt
```

On this 20-core host, default FineWeb shard splitting should use `15` parallel streams.

## RTX 5070 Tuned Profiles
- Tuned profile docs: `docs/RTX5070_TUNING.md`
- Saved JSON profiles:
  - `configs/train/rtx5070/fineweb_350bt_bpe_v2_longrun.json` (350BT long-run preset)
-  `configs/train/rtx5070/fineweb_350bt_bpe_v2_ctx1024_stage.json` (ctx1024 continuation preset)
- 350BT-first LR sweep (ctx 512, LR `2e-4..4e-4`):
```bash
bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh
```
- Reproducible context/batch benchmark sweep (tok/s + GPU memory/util summary):
```bash
bash scripts/benchmark_rtx5070_context_profiles.sh --max-steps 1200 --compile-model
```
- 350BT-first long run launcher:
```bash
bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh
```
- Auto-resume supervisor (refreshes manifest set between step chunks):
```bash
bash scripts/train_supervisor_rtx5070_350bt.sh \
  --step-chunk 2000 \
  --poll-seconds 60 \
  --batch-size 12 \
  --target-effective-batch 24 \
  --min-unique-input-files 510 \
  --sampler-strategy balanced \
  --sampler-min-full-passes 1 \
  --min-batch-size 6 \
  --max-batch-size 20 \
  --batch-step 2 \
  --generation-suite configs/eval/generation_smoke_suite_v1.json \
  --generation-every-chunks 1
```
- Phase-1 English conversation gating profile (before coding specialization):
```bash
bash scripts/train_supervisor_phase1_english_talk.sh
```
This uses `configs/eval/english_talk_suite_v1.json`,
`configs/eval/generation_talk_quality_v2.json`, and
`configs/eval/promotion_policy_talk_recovery_v2.json`.
It also runs a fixed holdout gate using
`configs/eval/english_talk_holdout_suite_v1.json`.
It also uses a dedicated state dir (`artifacts/reports/train_supervisor_phase1_talk`) and
lower-variance generation-gate settings (`--generation-temperature 0.2 --generation-top-k 1`).
Phase-1 launcher now defaults to recovery-friendly guardrails:
`--lr-schedule constant`, `--generation-fail-below-pass-rate 0.35`,
`--holdout-fail-below-pass-rate 0.35`, and `--promotion-min-quality-streak 2`.
Successful chunks update `artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt`,
which can be used to gate shard offload so only already-trained batches move to warm storage.
On each supervisor loop, hot-only manifest guard now runs automatically and disables any
active manifest that references symlinked shard bins.
Supervisor runs hot-shard warmup before each chunk and also a background warmup loop
during chunk training by default, hydrating missing active shard bins from Ceph into
hot storage (`scripts/hot_shard_warmup.py`) while GPU work is running.
When monitoring this profile, point status tools at that state dir:
`PYTHONPATH=src .venv/bin/python scripts/pipeline_live_view.py --supervisor-state-dir artifacts/reports/train_supervisor_phase1_talk`
and
`PYTHONPATH=src .venv/bin/python scripts/pipeline_eta_report.py --supervisor-state-dir artifacts/reports/train_supervisor_phase1_talk`.
If `--supervisor-state-dir` is omitted, both tools now auto-detect the newest existing
state dir between `artifacts/reports/train_supervisor_phase1_talk` and
`artifacts/reports/train_supervisor_350bt`.
For continuous 350BT ingestion/training, keep exactly one stage watchdog and one train supervisor running.
Avoid launching one-off `llm.cli train --max-steps ...` jobs in parallel with the supervisor.
Stage watchdog now performs stale worker cleanup before relaunch, so restarted controllers
do not leave orphan shard-build workers behind.
Supervisor now runs a manifest dedupe pass before each train chunk launch
(`scripts/fineweb_manifest_dedupe.py`, keep strategy `newest`) that disables exact duplicate
manifest file-sets and reports partial overlaps for review.
Use `--no-dedupe-overlap-manifests` to disable, or `--dedupe-dry-run` to audit without disabling duplicates.
Use `--dedupe-report-keep <N>` to cap saved dedupe report/log artifacts during long waits.
Use `--min-unique-input-files <N>` to hold training until enough unique parquet inputs are represented in manifests.
Use `--min-train-tokens <N>` to gate startup by total train-token coverage instead of raw file count.
Use `--lr-schedule constant` in supervisor for late-step recovery runs where cosine decay is too aggressive.
Use `--sampler-strategy balanced --sampler-min-full-passes <X>` to keep shard exposure
mixed and evenly distributed while guaranteeing minimum per-shard pass coverage each chunk.
Tune Ceph warm->hot hydration with `--hot-shard-warmup-workers <N>` and
`--hot-shard-warmup-max-files <N>` (or disable with `--no-hot-shard-warmup`).
Tune concurrent prefetch with `--hot-shard-warmup-background-interval-seconds <N>`,
`--hot-shard-warmup-background-max-files <N>`, or disable with
`--no-hot-shard-warmup-background`.
Supervisor enforces a singleton lock at
`artifacts/reports/train_supervisor_350bt/supervisor.lock`.
Add `--no-train-fail-on-eval-regression` if you want chunk runs to continue even when
the train-loop held-out perplexity gate is noisy; prompt-suite regression/promotion
checks still run in the supervisor eval step.
Supervisor resume guardrails now validate `last.pt`/`ckpt_step_*.pt` before resume and
quarantine invalid checkpoint files automatically, then continue from the newest valid one.
When post-chunk eval passes promotion logic (or beats prior pass-rate baseline), supervisor
also exports `best.pt`, `best_eval_report.json`, and safetensors best aliases.
Promotion discipline supports stricter gating by requiring eval policy promotion,
generation pass, holdout pass, and a minimum consecutive quality streak before best promotion.
Supervisor now updates `trained_batch_names.txt` from sampled shard traces produced by
`llm.cli train --sampled-shards-trace ...` (actual touched batches), not full manifest lists.
Supervisor can also auto-rollback to `best.pt` after sustained quality regressions; tune with
`--quality-rollback-streak <N>` and `--quality-rollback-cooldown-steps <N>`
(`0` streak disables rollback).
Supervisor outputs:
- `artifacts/reports/train_supervisor_350bt/train_trend.tsv` (per-chunk train telemetry)
- `artifacts/reports/train_supervisor_350bt/eval_trend.tsv` (post-chunk eval trend, including regression/promotion columns)
- `artifacts/reports/train_supervisor_350bt/generation_trend.tsv` (scheduled generation-gate trend, with regression columns)
- `artifacts/reports/train_supervisor_350bt/holdout_trend.tsv` (fixed holdout gate trend)
- `artifacts/reports/train_supervisor_350bt/eval_dashboard.html` (rendered trend dashboard)
- `artifacts/reports/train_supervisor_350bt/eval_dashboard_summary.json` (dashboard summary JSON)
The supervisor now auto-selects the latest successful eval baseline from the same suite
name/path as the active eval suite (and same behavior for generation-gate suite baselines),
so changing suites does not compare against mismatched historical reports.

Combined pipeline ETA/status reporter:
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_eta_report.py --loop --interval-seconds 60
```
Use `--once` for explicit single-snapshot mode (default behavior when `--loop` is not set).
Outputs:
- `artifacts/reports/pipeline_status.json`
- `artifacts/reports/pipeline_status.txt`
Includes embedded snapshots of `top -b -n1`, `free -h`, `nvidia-smi`, and `df -h`.
Also reports manifest coverage metrics (`manifest_unique_input_files`, overlap counts, `coverage_complete`).
Coverage metrics now include both active and offloaded manifests, so progress does not drop after safe offload.
Also reports hot-manifest metrics (`active_manifests`, `offloaded_manifests`,
`active_manifests_with_symlink_bins`, `trained_batch_names_count`).
Also reports `trainer_stall_seconds` and shard offload eligibility
(`offload_eligible_batches`, raw/capped counts, trained-registry presence).
Also reports `quality_heartbeat` (eval/gen/holdout trend state) and `status_confidence`
(`coverage`, `train_eta`, `quality`, `overall_score`).
Also includes per-task `RUN/STOP` state with stop reasons (for example `download complete`,
`staging handled by stage-loop`, `idle between chunks/eval`, or gate waits).
Task process counts are root-deduped (controller processes), so wrapper/child shells do not inflate `RUN xN`.

Live terminal view (single command to watch continuously):
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_live_view.py --refresh-seconds 5
```
This is a live-only monitor (no report/status files written) and includes:
- system status (CPU, memory, GPU, disk mounts)
- pipeline progress (download/staging/sharding/training)
- staging line includes `hot_parquet` and `hot_incomplete` to show active warm->hot copy progress
- hot-set status (`active_manifests`, `offloaded_manifests`, `active_symlink_manifests`, `trained_batches`)
- hot-set also shows shard offload readiness (`offload_eligible_batches`, raw eligible, cap)
- manifest coverage status (`unique/510`, overlap inputs/manifests, coverage rate + ETA, completion flag)
- supervisor gate status (for example waiting on `min_unique_input_files`)
- training row includes `stall=<seconds since last step progress>` for direct trainer stall visibility
- quality heartbeat line (`improving`/`flat`/`regressed`/`warming`) based on latest eval + generation trends
- confidence line (`coverage`, `train_eta`, `quality`, `overall`) to quickly judge status reliability
- running project task states with pid/runtime/cpu/mem summaries
- explicit stop reasons for tasks that are not running
- alert rows for stage-controller health and shard-manifest stall conditions
- training ETA fallback from `pipeline_status.json` (`--eta-status-file`) when live step deltas are temporarily flat

Coverage ETA/rate now falls back to sharding throughput when manifest overlap is zero, so
ETA remains visible between manifest update bursts.
Alerts also flag duplicate train controllers (`train-supervisor`/`trainer`) and unmanaged
stage-loop runs (stage-loop active without stage-watchdog).
Alerts also flag active manifests that still reference symlinked shard bins.
The train supervisor also self-checks process singleton by PID age within the same
`--state-dir` scope and exits newer duplicates, so accidental second launches do not persist.

It refreshes in-place (full-screen mode). If your terminal does not handle full-screen
escape codes well, add `--no-alt-screen`.

## Service Mode (systemd)
For reboot-safe long runs, install service units for supervisor + stage watchdog:

```bash
make install-systemd-services
```

No-sudo alternative (user units):
```bash
make install-user-systemd-services
```

If you launch supervisor as a transient user unit (`systemd-run --user`), set a high
open-files limit (for example `--property=LimitNOFILE=1048576`) so large shard sets
do not fail with `OSError: [Errno 24] Too many open files`.

Templates:
- `deploy/systemd/llm-train-supervisor.service`
- `deploy/systemd/llm-fineweb-stage-shard-loop.service`
- `deploy/systemd/llm-fineweb-stage-shard-watchdog.service`
- `deploy/systemd/llm-hf-download-watchdog.service`
- `deploy/systemd/llm-checkpoint-offload-prune.service`
- `deploy/systemd/llm-checkpoint-offload-prune.timer`
- `deploy/systemd/llm-bad-parquet-revalidate.service`
- `deploy/systemd/llm-bad-parquet-revalidate.timer`
- `deploy/systemd/llm-shard-offload.service`
- `deploy/systemd/llm-shard-offload.timer`
- `deploy/systemd/llm-vm-swappiness.service`
- user equivalents under `deploy/systemd/user/`

Revalidate and optionally restore bad parquet files:
```bash
PYTHONPATH=src .venv/bin/python scripts/revalidate_bad_parquet.py \
  --restage-valid \
  --max-entries 200 \
  --workers 8 \
  --max-restage-files 15 \
  --min-free-gib 80
```
Use `--max-entries` for incremental backlog cleanup; unprocessed entries stay on the bad list.
Use `--workers` to parallelize warm parquet validation on large bad-file backlogs.
This also prunes `artifacts/reports/fineweb_stage_shard_loop/quarantine_bad_parquet` by default:
- removes quarantine copies for files no longer marked bad
- for still-bad files, keeps only the newest copy per basename (`--quarantine-keep-per-name 1`)
- disable with `--no-prune-quarantine`

Offload older shard binaries to warm storage while keeping training hot-only:
```bash
PYTHONPATH=src .venv/bin/python scripts/offload_shard_bins_to_warm.py \
  --keep-local-batches 24 \
  --target-free-gib 180 \
  --max-batches 40 \
  --disable-offloaded-manifests \
  --require-trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt \
  --skip-if-trained-file-missing \
  --min-manifest-unique-input-files 510 \
  --min-active-manifests 48 \
  --min-active-train-tokens 40000000000
```
This replaces older local shard `.bin` files with warm-storage symlinks and renames
their `manifest.json` to `manifest.offloaded.json`, so `llm.cli train` only sees
local hot-disk manifests while disk usage stays bounded.
The `--require-trained-batches-file` guard prevents offloading any batch that has
not yet been included in a successful supervisor training chunk.
You can pass a comma-separated fallback list of trained-batch registry files
(for example phase1 + standard state dirs).
Use `--min-manifest-unique-input-files` to block offload until dataset coverage is complete.
Use `--min-active-manifests` and `--min-active-train-tokens` as offload
safety floor so hot-local training coverage never drops below your target.

Before offloading, reconcile previously offloaded manifests and re-enable any
batch not proven trained (plus optional hot rehydrate of active symlink bins):
```bash
PYTHONPATH=src .venv/bin/python scripts/reconcile_offloaded_manifests.py \
  --shards-root data/shards_global/fineweb-global-bpe-v1 \
  --trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt \
  --skip-if-trained-file-missing \
  --min-active-unique-input-files 510 \
  --rehydrate-active-symlink-bins
```

For timer automation, use the safe cycle wrapper
(reconcile -> offload -> reconcile -> enforce-hot-manifests):
```bash
bash scripts/shard_offload_cycle.sh
```

Environment template:
- `deploy/systemd/llm.env.example` (installed to `/etc/llm/llm.env`)
- Service units now pass through `LLM_*_ARGS` only when set; if unset, the underlying
  scripts use their built-in defaults.

Recommended `LLM_STAGE_SHARD_LOOP_ARGS` baseline for 20-core hosts:
```bash
LLM_STAGE_SHARD_LOOP_ARGS="--process-max-files 15 --shard-jobs 2 --auto-tune-shard-jobs --auto-tune-min-shard-jobs 2 --auto-tune-max-shard-jobs 3 --auto-tune-low-load-pct 80 --auto-tune-high-load-pct 95 --auto-tune-min-batch-seconds 300 --tokenizer-threads 10 --encode-batch-size 1024 --shard-size-tokens 20000000 --expected-unique-input-files 510 --coverage-complete-sleep-seconds 300 --sync-background --sync-max-inflight 2 --hot-max-used-pct 80 --offload-check-interval-seconds 120 --parquet-validate-timeout-seconds 180 --sleep-seconds 60 --shard-min-batch-size 512"
```
Recommended stage watchdog wrapper:
```bash
LLM_STAGE_SHARD_WATCHDOG_ARGS="--worker-args \"${LLM_STAGE_SHARD_LOOP_ARGS}\" --expected-unique-input-files 510 --check-interval-seconds 120 --stall-seconds 5400 --watchdog-log-file artifacts/reports/fineweb_stage_shard_loop/watchdog.log"
```
Recommended shard-offload cycle overrides:
```bash
LLM_SHARD_OFFLOAD_RECONCILE_ARGS="--shards-root data/shards_global/fineweb-global-bpe-v1 --trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt --skip-if-trained-file-missing --min-active-unique-input-files 510 --rehydrate-active-symlink-bins"
LLM_SHARD_OFFLOAD_ARGS="--shards-root data/shards_global/fineweb-global-bpe-v1 --warm-shards-root /mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1 --keep-local-batches 24 --target-free-gib 180 --max-batches 16 --disable-offloaded-manifests --require-trained-batches-file artifacts/reports/train_supervisor_phase1_talk/trained_batch_names.txt,artifacts/reports/train_supervisor_350bt/trained_batch_names.txt --skip-if-trained-file-missing --min-manifest-unique-input-files 510 --min-active-manifests 48 --min-active-train-tokens 40000000000"
```

## Inference Bundle Packaging
Build a portable local deploy bundle (with checksums and optional tarball):

```bash
PYTHONPATH=src .venv/bin/python scripts/package_inference_bundle.py \
  --checkpoint artifacts/checkpoints/fineweb-350bt-bpe-v2-run1/best.pt \
  --model-id local/fineweb-bpe-v2 \
  --create-tar
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
This now syncs training-critical inputs/outputs including:
`data/raw_zim`, `data/fineweb`, `data/cleaned`, `data/extracted`,
`data/shards`, `data/shards_global`, `artifacts/tokenizer`,
`artifacts/checkpoints`, and `artifacts/reports`.
- Periodic checkpoint offload + local prune:
```bash
bash scripts/checkpoint_offload_prune.sh \
  --local-checkpoints-dir artifacts/checkpoints \
  --warm-checkpoints-dir /mnt/ceph/llm/data/checkpoints \
  --keep-local-runs 1
```
- VM swappiness tuning (root):
```bash
sudo bash scripts/set_swappiness.sh --value 10 --persist
```
- Continuous ZIM offload worker (hot -> warm):
```bash
bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120
```
- Pull artifacts back from warm storage to local hot workspace:
```bash
bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data
```

## Current Capabilities
- Text stats CLI for quick corpus sanity checks.
- Batch corpus quality report generation (`corpus-quality-report`).
- Batch corpus cleanup and dedupe (`clean-corpus-batch`).
- Heuristic dataset risk auditing (`dataset-risk-report`).
- Direct FineWeb parquet -> tokenizer -> shard pipeline (`scripts/fineweb_parquet_to_shards.py`).
- BPE tokenizer workflow with train/save/load + contract fingerprinting.
- Token-window data pipeline (`TokenWindowDataset`) for next-token training pairs.
- ZIM archive text extraction (`extract-zim-text`) for server-hosted `.zim` files.
  - Automatically falls back to suggestion-index paths if fulltext search returns no matches.
- Corpus sharding (`shard-corpus`) into train/val token shard binaries + manifest.
- Batch corpus sharding (`shard-corpus-batch`) with one shared tokenizer.
- Baseline GPT training (`train`) with checkpoint save/resume.
  - Default architecture: RoPE + RMSNorm + SwiGLU (`gpt_rope_rmsnorm_swiglu_v1`).
  - Includes AdamW no-decay param groups, constant or warmup/cosine LR, and grad accumulation.
- Checkpoint-based text generation (`generate`) with temperature/top-k sampling.
- Optional safetensors export for deployment (`--export-safetensors`).
- Unit tests for tokenizer round-trips and unknown token behavior.

## Next Milestones
1. Expand checkpoint eval suite and track regressions in CI.
2. Add tokenizer-aware dataset manifests for long-running incremental FineWeb phases.
3. Raise context length beyond 1024 and benchmark memory/throughput tradeoffs.
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

Dataset inventory and intended use are tracked in:
- `wiki/Dataset-Registry.md`
