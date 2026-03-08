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
make test        # run unit tests
make lint        # run Ruff checks
make format      # run Black formatter
make typecheck   # run MyPy
make smoke       # tiny CLI smoke check
make verify-shards # print shard integrity check usage
make train       # print baseline training command usage
make generate    # print checkpoint text-generation command usage
make eval-checkpoint # print standardized prompt-suite eval usage
make train-tokenizer-global # print shared-tokenizer command usage
make corpus-quality-report # print quality report command usage
make clean-corpus-batch # print batch cleanup command usage
make dataset-risk-report # print heuristic dataset risk audit command usage
make pull-hf-rows # print Hugging Face rows API pull helper usage
make fineweb-parquet-to-shards # print direct FineWeb parquet->token-shards usage
make stage-fineweb-from-warm # print warm->hot FineWeb chunk staging usage
make fineweb-stage-shard-loop # print rolling stage->shard->verify->sync->purge usage
make fineweb-hot-queue # print hot parquet queue-oriented stage/shard usage
make lr-sweep-350bt # print RTX 5070 LR sweep usage for staged 350BT shards
make train-350bt-v2 # print 350BT long-run launcher usage
make train-supervisor-350bt # print auto-resume trainer supervisor usage
make pipeline-eta # print combined download/shard/train ETA reporter usage
make pipeline-live # print live terminal pipeline dashboard usage
make shard-corpus-batch # print shared-tokenizer batch sharding usage
make hf-download-resumable # print self-healing HF resume-download worker usage
make sync-warm   # sync raw/training data + artifacts to warm storage
make hydrate-warm # hydrate hot workspace from warm storage
make offload-zim # continuously move raw ZIMs hot -> warm
make hf-prepare-publish # print HF bundle/publish usage
make hf-download-model # print full HF model download usage
make serve-openai # print local OpenAI-compatible server usage
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
For talking-only passes, keep code filtering enabled (default) or tune with:
`--code-symbol-ratio-threshold` and `--code-keyword-hits-threshold`.

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

3ab. Stage FineWeb chunks from warm to hot as needed:
```bash
bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8
```

3ac. Run rolling warm->hot staging + sharding loop (recommended for 350BT on limited hot disk):
```bash
bash scripts/fineweb_stage_shard_loop.sh \
  --hot-queue-min-files 8 \
  --stage-max-files 2 \
  --process-max-files 4 \
  --sleep-seconds 60 \
  --shard-min-batch-size 512
```
This loop stages bounded parquet files to hot storage, builds verified shard batches under
`data/shards_global/fineweb-global-bpe-v1/`, syncs those batches back to warm storage,
and purges processed hot parquet files.
`--hot-queue-min-files` keeps a small parquet queue staged locally so shard building is less likely to idle on copy waits.
If a shard build fails with OOM-like errors, the loop retries automatically with a smaller batch size.

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
Training now supports:
- warmup + cosine LR schedule (`--lr-schedule`, `--lr-warmup-steps`, `--lr-min-ratio`)
- gradient accumulation (`--grad-accum-steps`)
- fixed held-out eval batches (`--no-eval-freeze-batches` to disable)
- eval regression gate (`--fail-on-eval-regression --eval-regression-tolerance 0.20`)
- optional weights-only export (`--export-safetensors`)

9. Generate text from a checkpoint:
```bash
PYTHONPATH=src .venv/bin/python -m llm.cli generate \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/last.pt \
  --prompt "The future of medicine is" \
  --max-new-tokens 200 \
  --temperature 0.9 \
  --top-k 50
```

10. Run standardized checkpoint eval (fixed prompt suite + scored report):
```bash
PYTHONPATH=src .venv/bin/python scripts/eval_checkpoint_prompts.py \
  --checkpoint artifacts/checkpoints/medlineplus_baseline/last.pt \
  --suite configs/eval/standard_prompt_suite_v1.json
```
Writes a JSON report under `artifacts/reports/evals/` so runs can be compared over time.

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

Optional text-first path still exists for inspection-heavy runs:
`parquet_to_corpus -> clean-corpus-batch -> train-tokenizer-global -> shard-corpus-batch`.

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
  - `configs/train/rtx5070/fineweb_global_bpe_v1_big.json` (recommended, BPE)
  - `configs/train/rtx5070/fineweb_350bt_bpe_v2_longrun.json` (350BT long-run preset)
- Launch tuned big profile:
```bash
bash scripts/train_rtx5070_fineweb_bpe_v1_big.sh
```
- 350BT-first LR sweep (ctx 512, LR `2e-4..4e-4`):
```bash
bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh
```
- 350BT-first long run launcher:
```bash
bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh
```
- Auto-resume supervisor (refreshes manifest set between step chunks):
```bash
bash scripts/train_supervisor_rtx5070_350bt.sh \
  --step-chunk 2000 \
  --poll-seconds 120 \
  --target-effective-batch 34
```
Supervisor outputs:
- `artifacts/reports/train_supervisor_350bt/train_trend.tsv` (per-chunk train telemetry)
- `artifacts/reports/train_supervisor_350bt/eval_trend.tsv` (post-chunk eval trend)

Combined pipeline ETA/status reporter:
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_eta_report.py --loop --interval-seconds 60
```
Outputs:
- `artifacts/reports/pipeline_status.json`
- `artifacts/reports/pipeline_status.txt`
Includes embedded snapshots of `top -b -n1`, `free -h`, `nvidia-smi`, and `df -h`.

Live terminal view (single command to watch continuously):
```bash
PYTHONPATH=src .venv/bin/python scripts/pipeline_live_view.py --refresh-seconds 5
```
The live view auto-fits to terminal size and refreshes in-place (full-screen mode).
If your terminal does not handle full-screen escape codes well, add `--no-alt-screen`.

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
  - Includes AdamW no-decay param groups, warmup/cosine LR, and grad accumulation.
- Checkpoint-based text generation (`generate`) with temperature/top-k sampling.
- Optional safetensors export for deployment (`--export-safetensors`).
- Unit tests for tokenizer round-trips and unknown token behavior.

## Next Milestones
1. Expand checkpoint eval suite and track regressions in CI.
2. Add tokenizer-aware dataset manifests for long-running incremental FineWeb phases.
3. Add larger-context training profiles and memory/throughput benchmarking.
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
