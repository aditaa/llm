# Repository Guidelines

## Project Structure & Module Organization
The codebase is a Python LLM-from-scratch scaffold.
- `src/llm/`: core package (`tokenizer.py`, `data.py`, `sharding.py`, `cli.py`, `model.py`)
- `tests/`: unit tests (tokenizer + data pipeline + sharding coverage)
- `docs/`: architecture and roadmap docs
- `information/`: references, imported notes, and source material
- `data/`: local/intermediate corpora (gitignored except `data/README.md`)
- `artifacts/`: generated outputs such as vocab/checkpoints (gitignored)

Keep modules single-purpose and expand by domain (for example: `src/llm/training.py`, `src/llm/data.py`).

## Build, Test, and Development Commands
Use the `Makefile` as the source of truth:
- `make setup-dev`: create `.venv`, install dev deps, init submodules
- `make setup-train`: install training/notebook extras
- `make setup-infer`: install inference/deploy extras
- `make install-server-system`: install Ubuntu/Debian system packages
- `make install-systemd-services`: install/reload systemd service units for long-running workers
- `make install-user-systemd-services`: install/reload user-level systemd service units (no sudo path; includes shard/checkpoint timers)
- `make doctor`: environment/tooling diagnostics
- `make test`: run `unittest` test suite
- `make lint`: run Ruff lint checks
- `make format`: run Black formatter
- `make typecheck`: run MyPy on `src/`
- `make smoke`: run a minimal CLI smoke test
- `make verify-shards`: usage helper for shard integrity verification
- `make train`: usage helper for baseline GPT training
- `make generate`: usage helper for checkpoint text generation
- `make average-checkpoints`: usage helper for checkpoint weight averaging
- `make eval-checkpoint`: usage helper for standardized checkpoint prompt-suite eval
- `make render-eval-dashboard`: usage helper for rendering eval trend HTML/JSON dashboard
- `make package-inference-bundle`: usage helper for local deploy bundle packaging with checksums
- `make train-tokenizer-global`: usage helper for shared tokenizer training
- `make corpus-quality-report`: usage helper for corpus quality scan
- `make clean-corpus-batch`: usage helper for batch corpus cleanup
- `make dataset-risk-report`: usage helper for heuristic dataset risk audit
- `make pull-hf-rows`: usage helper for bounded Hugging Face rows pulls
- `make fineweb-parquet-to-shards`: usage helper for direct FineWeb parquet -> tokenizer -> shard conversion
- `make fineweb-manifest-dedupe`: usage helper for overlap-manifest dedupe audit/fix
- `make stage-fineweb-from-warm`: usage helper for staging FineWeb parquet chunks from warm to hot
- `make fineweb-revalidate-bad-parquet`: usage helper for revalidating/restaging bad parquet entries
- `make reconcile-offloaded-manifests`: usage helper for restoring risky offloaded manifests and optional bin rehydrate
- `make shard-offload-cycle`: usage helper for safe reconcile -> offload -> reconcile timer cycle
- `make offload-shard-bins-warm`: usage helper for replacing older local shard `.bin` files with warm-storage symlinks, disabling offloaded manifests, gating offload by trained-batch registry, and honoring hot-coverage safety floors
- `make hot-shard-warmup`: usage helper for hydrating active shard bins from warm to hot storage (pre-chunk or background prefetch)
- `make enforce-hot-manifests`: usage helper for disabling active manifests that reference symlinked shard bins
- `make fineweb-stage-shard-loop`: usage helper for rolling warm->hot stage + shard + verify + sync + purge
- `make fineweb-stage-shard-watchdog`: usage helper for auto-restart watchdog around the stage/shard loop
- `make lr-sweep-350bt`: usage helper for RTX 5070 LR sweep on staged 350BT shards (`2e-4..4e-4`, ctx 512)
- `make benchmark-rtx5070`: usage helper for short context/batch throughput + GPU memory benchmark sweep
- `make train-350bt-v2`: usage helper for the 350BT long-run launcher profile
- `make train-350bt-ctx1024`: usage helper for context-extension continuation stage
- `make train-supervisor-350bt`: usage helper for auto-resume chunked training that refreshes manifest set between cycles
- `make train-supervisor-phase1-talk`: usage helper for phase-1 English conversation gating profile
- `make pipeline-eta`: usage helper for combined download + sharding + training ETA/status reporting
- `make pipeline-live`: usage helper for a live terminal pipeline dashboard
- `make shard-corpus-batch`: usage helper for batch sharding with a shared tokenizer
- `make checkpoint-offload-prune`: usage helper for checkpoint warm sync + local prune policy
- `make checkpoint-step-offload`: usage helper for offloading older `ckpt_step_*.pt` files to warm storage while preserving newest local resume steps
- `make set-swappiness`: usage helper for host swappiness tuning (root)
- `make hf-download-resumable`: usage helper for self-healing Hugging Face resume-download worker
- `make hf-download-watchdog`: usage helper for watchdog auto-restart around stalled/exited HF download workers
- `make hf-prepare-publish`: usage helper for Hugging Face release bundle/publish
- `make hf-download-model`: usage helper for full Hugging Face model snapshot download
- `make serve-openai`: usage helper for local OpenAI-compatible serving

Server setup reference:
`docs/SERVER_SETUP.md`

CI/CD workflows:
- `.github/workflows/ci.yml`: script sanity (`bash -n` + `py_compile`), lint, typecheck, tests, smoke, and gate job
- `.github/workflows/wiki-sync.yml`: publishes wiki pages on `main` doc changes
- `.github/dependabot.yml`: weekly dependency update PRs for `pip`, `requirements/`, and GitHub Actions
- maintenance units: `deploy/systemd/llm-checkpoint-offload-prune.service` + `.timer`, `deploy/systemd/llm-checkpoint-step-offload.service` + `.timer`, `deploy/systemd/llm-bad-parquet-revalidate.service` + `.timer`, `deploy/systemd/llm-shard-offload.service` + `.timer`, `deploy/systemd/llm-vm-swappiness.service`

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, UTF-8 files
- Max line length: 100 (Black/Ruff configured in `pyproject.toml`)
- Use `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_CASE` for constants
- Add type hints for public functions; MyPy is configured with `disallow_untyped_defs = true`

## Testing Guidelines
- Framework: `unittest` (`tests/test_*.py`)
- Mirror source layout when adding tests (example: `src/llm/data.py` -> `tests/test_data.py`)
- Cover edge cases and failure modes, not only happy paths
- Run `make test` locally before each commit
- Keep `make smoke` torch-independent in CI (non-training CLI commands should not require `torch` import at module load)

## Commit & Pull Request Guidelines
Follow concise, imperative commit subjects (<= 72 chars), for example:
- `Add tokenizer save/load round-trip tests`
- `Implement corpus stats CLI command`

PRs should include:
- What changed and why
- How it was validated (`make test`, `make lint`, sample output if relevant)
- Linked issue/ticket when applicable

Keep PR scope narrow; split refactors and features into separate PRs.
`main` should be protected with required status check `CI Gate`.

## Security & Configuration Tips
- Never commit secrets or credentials
- Never commit raw dataset dumps or `.zim` archives
- Never commit copyrighted book/PDF source files
- Keep generated files in `artifacts/` and out of git history
- Prefer environment variables for machine-specific settings
- Run all processing in local hot workspace (`./data`, `./artifacts`)
- Use warm storage at `/mnt/ceph/llm/data` as cache/backup
- For first-pass talking-only selection + hot/warm rebalance, run:
  `bash scripts/first_pass_zim_profile.sh --move-excluded`
- Run shard integrity verification before training (`llm.cli verify-shards`)
- `extract-zim-text` now falls back to suggestion-index paths when fulltext search has zero matches
- If extraction still returns `written_articles=0`, retry with a lower `--min-chars` (for example `20`)
- For ZIMs without fulltext index, generate a paths list from suggestion/title index and run `extract-zim-text --paths-file ...`
- `llm.cli train` requires a tokenizer-compatible shard set (same tokenizer mapping across all selected manifests)
- Shard manifests now record `tokenizer_hash` + `tokenizer_contract_hash`; do not mix manifests with mismatched values
- Preferred multi-dataset flow: `train-tokenizer-global` -> `shard-corpus-batch` -> `train`
- Preferred pre-tokenization flow: `corpus-quality-report` -> `clean-corpus-batch` -> `train-tokenizer-global`
- Run `dataset-risk-report` on cleaned corpora before tokenizer training and manually review flagged slices
- Prefer `clean-corpus-batch --en-only` for first-pass talking runs
- For talking-only runs, keep `clean-corpus-batch` code-like filtering enabled (default)
- Cleanup defaults also enforce min words, URL density, symbol density, and repetitive-token filters
- Cleanup defaults also apply normalized dedupe keys and contamination filtering; tune with `--no-dedupe-normalized`, `--no-drop-contamination`, `--contamination-pattern`, and `--contamination-patterns-file`
- Use `bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data` to copy local artifacts to warm storage
- `sync_warm_storage.sh` covers raw + training data: `data/raw_zim`, `data/fineweb`, `data/cleaned`, `data/extracted`, `data/shards`, `data/shards_global`, `artifacts/tokenizer`, `artifacts/checkpoints`, `artifacts/reports`
- Use `bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120` for continuous hot->warm raw ZIM offload
- Use `bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data` to restore local artifacts from warm storage
- Use `python3 scripts/offload_shard_bins_to_warm.py --disable-offloaded-manifests --require-trained-batches-file <phase1_state>/trained_batch_names.txt,<standard_state>/trained_batch_names.txt --skip-if-trained-file-missing --min-manifest-unique-input-files <N> --min-active-manifests <N> --min-active-train-tokens <TOKENS>` to move only already-trained older shard bins to warm storage while keeping active manifests hot-local only
- Use `python3 scripts/reconcile_offloaded_manifests.py --trained-batches-file <phase1_state>/trained_batch_names.txt,<standard_state>/trained_batch_names.txt --skip-if-trained-file-missing --min-active-unique-input-files <N> --max-restore <N> --warm-shards-root /mnt/ceph/llm/data/shards_global/fineweb-global-bpe-v1 --rehydrate-restored-bins --rehydrate-active-symlink-bins` before offload runs to restore untrained/under-coverage offloaded manifests and rehydrate restored/active bins back to hot storage
- Prefer `bash scripts/shard_offload_cycle.sh` in automation (pre-reconcile -> gated offload -> post-reconcile -> enforce-hot-manifests) instead of direct one-shot offload calls
- For bounded external pulls (for example FineWeb samples), use `python3 scripts/pull_hf_rows.py` and write to warm storage first
- For long-running Hugging Face parquet pulls, use `scripts/hf_download_resumable.sh` instead of one-shot `hf download` (prefer `--enable-hf-transfer`, `--max-workers 6`, `--skip-dry-run`, and `--attempt-timeout-seconds` for 350BT-scale pulls)
- For unattended long pulls, prefer `scripts/hf_download_watchdog.sh` to restart stalled/exited resumable workers based on progress checks (`--stall-seconds`, `--check-interval-seconds`)
- For watchdog runs, set `--exit-on-complete` with `--expected-parquet-files` and/or `--expected-bytes` so completed pulls do not restart indefinitely
- `hf_download_watchdog.sh` enforces a singleton lock (`.hf_download_watchdog.lock`) in the target local-dir
- For parquet-based FineWeb workflows, use `scripts/stage_fineweb_from_warm.sh` to copy bounded warm chunks into hot storage
- `stage_fineweb_from_warm.sh` writes to `*.parquet.incomplete` and atomically renames to `.parquet` after size validation
- Use `stage_fineweb_from_warm.sh --copy-jobs <N>` to parallelize warm->hot staging copies
- Use `stage_fineweb_from_warm.sh --min-free-gib <N>` to keep a hot-disk free-space floor during staging
- Use `stage_fineweb_from_warm.sh --skip-list <bad_file_list>` to avoid re-staging known-bad parquet basenames
- For long-running 350BT ingestion, use `scripts/fineweb_stage_shard_loop.sh` in direct-source mode (default: read parquet from Ceph path directly) and keep shard outputs hot-local
- `fineweb_stage_shard_loop.sh` supports automatic training-focused mode after full coverage (`--expected-unique-input-files 510` + default coverage-complete mode) to pause staging/sharding churn once all inputs are represented
- For unattended long 350BT ingestion, run `scripts/fineweb_stage_shard_watchdog.sh` to auto-restart stage-loop worker exits/stalls
- `fineweb_stage_shard_watchdog.sh` now holds worker restarts when manifest unique-input coverage reaches target (`--expected-unique-input-files`)
- For long shard batches, use a higher stage-watchdog stall timeout (for example `--stall-seconds 5400`) to avoid false restarts mid-batch
- `fineweb_stage_shard_watchdog.sh` enforces a singleton lock in the stage state dir (`watchdog.lock`) and stops worker process groups (not only the parent shell)
- `fineweb_stage_shard_watchdog.sh` progress snapshot includes hot `.incomplete` file count/bytes so active warm->hot copy phases are not treated as stalls
- `fineweb_stage_shard_watchdog.sh` now also cleans stale stage-loop/shard-worker processes before relaunch so only one controller remains active
- `fineweb_stage_shard_watchdog.sh` can adopt an already-running stage-loop controller (default) so watchdog restarts do not leave direct loop runs unmanaged; use `--no-adopt-existing-loop` to force fresh worker launch
- Use `fineweb_stage_shard_loop.sh --enable-stage-copy --hot-queue-min-files <N>` only when you explicitly want legacy warm->hot parquet staging
- `stage_fineweb_from_warm.sh` now uses a per-destination lock so concurrent staging calls serialize safely
- `stage_fineweb_from_warm.sh --lock-wait-seconds 0` (default) skips quickly when another staging call holds the lock; tune if you want blocking behavior
- `stage_fineweb_from_warm.sh` only applies rsync `--contimeout` for rsync-daemon sources (`rsync://` or `::`); local/NFS paths avoid this flag
- Use `fineweb_stage_shard_loop.sh --enable-stage-copy --stage-copy-jobs <N>` to forward parallel staging copy workers into each stage cycle
- Use `fineweb_stage_shard_loop.sh --enable-stage-copy --no-auto-tune-stage-copy-jobs` to pin copy workers, or keep default copy auto-tune enabled (`--auto-tune-min-copy-jobs`, `--auto-tune-max-copy-jobs`, `--auto-tune-iowait-low-pct`, `--auto-tune-iowait-high-pct`) for queue/iowait-driven staging throughput control
- Use `fineweb_stage_shard_loop.sh --enable-stage-copy --stage-min-free-gib <N>` so staging never drives hot disk below a free-space guardrail
- Use `fineweb_stage_shard_loop.sh --auto-tune-shard-jobs` to adapt shard parallelism and tokenizer threads from CPU load + batch runtime
- Use `fineweb_stage_shard_loop.sh --sync-background --sync-max-inflight <N>` to overlap warm sync with next shard batches and reduce idle wait
- Keep hot disk near target by enabling loop auto-offload (`--hot-max-used-pct 95` + `--offload-check-interval-seconds 120`), which offloads already-trained older shard batches without pausing sharding
- Prefer larger shard files for throughput (`--shard-size-tokens 20000000` for FineWeb 350BT pipeline)
- `fineweb_stage_shard_loop.sh` now drains background sync jobs on `INT`/`TERM` for safer restarts
- Default systemd loop/watchdog templates now use stage free-space guardrails + auto-tune; override with `LLM_STAGE_SHARD_LOOP_ARGS` in `/etc/llm/llm.env` if needed
- Systemd service units pass through `LLM_*_ARGS` only when set; otherwise, script-level defaults apply
- `fineweb_stage_shard_loop.sh` now preflights selected parquet files and quarantines failures into `artifacts/reports/fineweb_stage_shard_loop/quarantine_bad_parquet/`
- Known-bad parquet basenames are tracked in `artifacts/reports/fineweb_stage_shard_loop/bad_parquet_files.txt` and skipped in future stage cycles
- On shard-build non-OOM corruption failures, `fineweb_stage_shard_loop.sh` attempts one warm->hot restage retry in stage-copy mode before quarantining inputs as bad
- Stage-loop deep-validates newly staged parquet files in stage-copy mode (`--deep-validate-max-batches`, `--deep-validate-batch-size`) and quarantines early failures as `stage_deep_validation_failed`
- Use `--parquet-validate-timeout-seconds <N>` to bound Ceph parquet validation/deep-validation calls and avoid startup hangs on single files
- On startup, `fineweb_stage_shard_loop.sh` reconciles bad parquet entries against warm-source validity to avoid permanent false-positive skips
- `fineweb_stage_shard_loop.sh` now bootstraps processed parquet basenames from existing manifests at startup, merges `processed + bad` into a stage skip list, and removes known files from hot storage before staging
- Stage-loop batch guardrails now require valid report + manifest + non-empty shard files before marking files as processed/purging hot copies
- Guardrail validation logic is centralized in `src/llm/fineweb_guardrails.py`; keep it covered by unit tests
- For higher CPU throughput on this 20-core host, prefer `--shard-jobs 2 --tokenizer-threads 10 --encode-batch-size 1024`
- Keep stage-loop OOM retry enabled (default) so shard builds back off to smaller `--batch-size` automatically
- For continuously growing shard sets, use `scripts/train_supervisor_rtx5070_350bt.sh` so each resumed chunk re-reads new manifests before training continues
- For staged 350BT training, set `train_supervisor_rtx5070_350bt.sh --min-unique-input-files <N>` (for example `510`) to avoid starting on too-small shard coverage
- Alternatively gate on coverage by token volume with `train_supervisor_rtx5070_350bt.sh --min-train-tokens <N>`
- Supervisor now runs `scripts/fineweb_manifest_dedupe.py` before each train chunk launch to disable exact duplicate manifest file-sets (`keep=newest`) and report partial overlaps; use `--no-dedupe-overlap-manifests` or `--dedupe-dry-run` as needed
- Use supervisor `--dedupe-report-keep <N>` to cap per-chunk dedupe report/log file growth during long coverage waits
- For continuous 350BT pipeline runs, keep exactly one `fineweb_stage_shard_watchdog.sh` and one `train_supervisor_rtx5070_350bt.sh` process active; avoid concurrent one-off `llm.cli train` runs against the same checkpoint directory
- `train_supervisor_rtx5070_350bt.sh` now acquires `artifacts/reports/train_supervisor_350bt/supervisor.lock` directly for non-bypassable singleton control
- `train_supervisor_rtx5070_350bt.sh` also self-demotes newer duplicate supervisor shells by PID age within the same `--state-dir` scope, so accidental second launches exit cleanly
- When launching supervisor via transient `systemd-run --user`, set `--property=LimitNOFILE=1048576` to avoid `OSError: [Errno 24] Too many open files` on large shard sets
- Supervisor resume guardrail validates `last.pt`/`ckpt_step_*.pt` and quarantines invalid checkpoint files before retry
- Use `--no-train-fail-on-eval-regression` in supervisor when you want train chunks to continue and rely on post-chunk prompt-suite gates
- Supervisor baseline selection now matches the active suite (`suite_name`/`suite_path`) for both eval and generation gates so suite changes do not compare against mismatched historical reports
- For phase-1 talking quality, use `scripts/train_supervisor_phase1_english_talk.sh` (`english_talk_suite_v1` + `generation_talk_quality_v2` + fixed `english_talk_holdout_suite_v1`) before code-specialization passes
- Phase-1 launcher now defaults to recovery-oriented quality gates (`--lr-schedule constant`, `--generation-fail-below-pass-rate 0.35`, `--holdout-fail-below-pass-rate 0.35`, `--promotion-min-quality-streak 2`, `promotion_policy_talk_recovery_v2.json`)
- Phase-1 launcher writes supervisor state to `artifacts/reports/train_supervisor_phase1_talk`; pipeline status tools now auto-detect between phase1 + standard state dirs (override with `--supervisor-state-dir` when needed)
- Phase-1 launcher uses lower-variance generation gating (`--generation-temperature 0.2 --generation-top-k 1`)
- Supervisor now runs hot-manifest guard each loop (`scripts/enforce_hot_only_manifests.py`) to auto-disable active manifests that reference symlinked shard bins
- Supervisor now supports hot-shard warmup (`scripts/hot_shard_warmup.py`) both pre-chunk and in background during train chunks to hydrate missing/symlinked active shard bins from Ceph into hot storage (`--hot-shard-warmup-workers`, `--hot-shard-warmup-max-files`, `--hot-shard-warmup-background-interval-seconds`, `--hot-shard-warmup-background-max-files`, `--no-hot-shard-warmup`, `--no-hot-shard-warmup-background`)
- Prefer supervisor `--sampler-strategy balanced --sampler-min-full-passes <X>` for even shard mixing with guaranteed minimum per-shard full-pass coverage per chunk
- On 12 GB RTX 5070 profiles, start supervisor with `--batch-size 12 --target-effective-batch 24 --min-batch-size 6 --max-batch-size 20 --batch-step 2` to avoid early OOM churn
- Use supervisor `--train-stall-check-seconds` + `--train-stall-kill-seconds` to auto-restart stuck train chunks when step progress stops
- Supervisor writes chunk trends to `artifacts/reports/train_supervisor_350bt/train_trend.tsv` and post-chunk eval trends to `artifacts/reports/train_supervisor_350bt/eval_trend.tsv`
- Supervisor also writes scheduled generation-gate trends to `artifacts/reports/train_supervisor_350bt/generation_trend.tsv`
- Supervisor also writes fixed holdout-gate trends to `artifacts/reports/train_supervisor_350bt/holdout_trend.tsv` and stores frozen baseline path in `holdout_baseline_report.txt`
- Supervisor also renders `artifacts/reports/train_supervisor_350bt/eval_dashboard.html` and exports `best.pt` aliases after successful eval promotions
- Supervisor now records successful-chunk sampled-batch coverage in `<state_dir>/trained_batch_names.txt` (from `llm.cli train --sampled-shards-trace`) for safe shard offload gating
- Supervisor supports quality auto-rollback to `best.pt` after sustained regressions; tune with `--quality-rollback-streak` and `--quality-rollback-cooldown-steps`
- Use `--generation-suite configs/eval/generation_smoke_suite_v1.json` (or `generation_talk_quality_v2.json` for phase-1 talk quality) and `--generation-every-chunks <N>` to run prompt-generation drift gates every chunk (or every N chunks)
- Use `scripts/pipeline_eta_report.py --loop` for combined ETA snapshots in `artifacts/reports/pipeline_status.{json,txt}` (includes `top`, `free -h`, `nvidia-smi`, and `df -h` captures)
- `pipeline_eta_report.py` accepts `--once` for explicit single-snapshot mode
- `pipeline_eta_report.py` also tracks manifest coverage quality (`manifest_unique_input_files`, overlap counts, `coverage_complete`)
- `pipeline_eta_report.py` now includes per-task `RUN/STOP` state with stop reasons and `supervisor_gate` in JSON/text output
- `pipeline_eta_report.py` now also reports `trainer_stall_seconds` and shard offload readiness (`offload_eligible_batches`, raw/capped counts)
- `pipeline_eta_report.py` now also exports `quality_heartbeat` (eval + generation + holdout) plus `status_confidence` (`coverage`, `train_eta`, `quality`, `overall_score`)
- Use `scripts/pipeline_live_view.py --refresh-seconds 5` for a live-only terminal monitor (system + pipeline task status, no report writes; includes watchdog/stage-loop/generation-gate task rows; add `--no-alt-screen` if needed)
- `pipeline_live_view.py` can reuse fresh ETA report train-step rates via `--eta-status-file`/`--eta-status-max-age-seconds` to keep training ETA visible when live step deltas are flat
- `pipeline_live_view.py` staging line includes `hot_parquet` + `hot_incomplete` so cache refill progress is visible during warm->hot copies
- `pipeline_live_view.py` includes manifest coverage line (`unique/510`, overlap inputs/manifests, completion flag)
- `pipeline_live_view.py` also shows hot-manifest state (`active`, `offloaded`, `active_symlink_manifests`) and trained-batch registry count
- `pipeline_live_view.py` now also shows shard offload readiness (`offload_eligible_batches`) and training stall age (`stall=<seconds>`)
- `pipeline_live_view.py` now shows a quality heartbeat (`improving`/`flat`/`regressed`/`warming`) from latest eval + generation + holdout trend files
- `pipeline_live_view.py` now also shows a confidence row (`coverage`, `train_eta`, `quality`, `overall`) to gauge status reliability
- `pipeline_live_view.py` includes manifest coverage rate/ETA to gauge when coverage gates will clear
- `pipeline_live_view.py` also shows supervisor gate state (for example `waiting_unique_inputs <have>/<need>` or `waiting_train_tokens <have_tokens>/<need_tokens>`)
- `pipeline_live_view.py` coverage ETA/rate falls back to sharding throughput when manifest overlap is zero, so ETA remains visible between manifest-update bursts
- `pipeline_live_view.py` alerts on duplicate train controllers and on stage-loop runs that are not watchdog-managed
- `pipeline_eta_report.py` task process counters are root-deduped so wrapper/child shells do not inflate `RUN xN` values
- Revalidate/recover bad parquet list entries with `scripts/revalidate_bad_parquet.py`; use `--max-entries` for incremental backlog cleanup, `--workers` for parallel validation, and `--restage-valid` to copy newly validated files back into hot storage
- `revalidate_bad_parquet.py` also prunes `quarantine_bad_parquet` by default (drops stale/validated entries; keeps newest copy per still-bad basename)
- Optional automation: `llm-bad-parquet-revalidate.timer` runs `revalidate_bad_parquet.py` periodically via systemd
- For checkpoint regression tracking, run `scripts/eval_checkpoint_prompts.py` with `configs/eval/standard_prompt_suite_v3.json`; use `--baseline-report` and `--promotion-policy configs/eval/promotion_policy_v1.json` to emit regression deltas + promotion verdict
- Promotion/comparison logic lives in `src/llm/eval_policy.py`; keep policy checks unit-tested (`tests/test_eval_policy.py`)
- For FineWeb-first training runs, build shards directly with `PYTHONPATH=src .venv/bin/python scripts/fineweb_parquet_to_shards.py --input-dir data/fineweb/sample-350BT --output-dir data/shards_global/fineweb-global-bpe-v1 --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json --bpe-vocab-size 32000 --field text`
- FineWeb-only baseline flow: `fineweb_parquet_to_shards -> verify-shards -> train`
- For incremental FineWeb adds, freeze tokenizer on phase1 and build later phases with `--tokenizer-in` plus `--files-list`; resume training from `last.pt` with same `--shards-path` root
- On this 20-core server, use 15 parallel streams for split shard-build runs
- For CUDA training throughput, prefer `llm.cli train --precision auto` (disable TF32 only if needed with `--no-tf32`)
- If GPU utilization stays bursty, try `llm.cli train --compile-model --compile-mode reduce-overhead`
- `llm.cli train --compile-model` now warms the compiled graph and falls back to eager by default; add `--compile-strict` to hard-fail on compile init/warmup issues
- Use `llm.cli train --sampler-max-open-shards <N>` to cap open shard memmaps and reduce file-descriptor pressure
- Default training architecture is `gpt_rope_rmsnorm_swiglu_v1`; use legacy profile only for old checkpoint compatibility
- Prefer `llm.cli train --lr-schedule cosine --lr-warmup-steps <N>` for fresh first-pass runs; prefer supervisor `--lr-schedule constant` for late-step recovery when cosine decay has already flattened LR
- Use `--grad-accum-steps` when VRAM is tight and you need higher effective batch
- Keep disk use bounded with `llm.cli train --checkpoint-keep-last <N> --checkpoint-keep-every <M>`
- Use `scripts/checkpoint_offload_prune.sh` to sync checkpoint runs to warm storage and prune older local runs (keep active + newest local)
- Use `scripts/checkpoint_step_offload.sh` to offload older `ckpt_step_*.pt` files from active runs to warm storage on a short cadence (for example every 10 minutes) while keeping newest local resume checkpoints
- Tune host swap behavior with `sudo bash scripts/set_swappiness.sh --value 10 --persist` (or set `LLM_SWAPPINESS=10` in `/etc/llm/llm.env` for systemd)
- For context-extension continuation, resume with `llm.cli train --allow-context-extension --context-length 1024 ...`
- Use EMA for long runs with `--ema-decay 0.999 --ema-start-step <warmup_end>` and generate with `--use-ema` when present
- Keep held-out eval batches frozen (default) and enable regression gating with `--fail-on-eval-regression`
- Optimizer uses no-weight-decay groups for norms/biases/embeddings by default
- For post-run smoothing, merge several checkpoints with `llm.cli average-checkpoints --state-key model_state` (or `ema_state`)
- For deploy bundles, use `scripts/package_inference_bundle.py` (checksums + optional tarball) or `hf_prepare_and_publish_model.py --include-safetensors`
- RTX 5070 tuned training profiles live in `configs/train/rtx5070/`; preferred 350BT launchers: `bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh` then `bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh`
- For reproducible throughput/memory probes across context lengths, use `bash scripts/benchmark_rtx5070_context_profiles.sh` and track `summary.tsv` outputs under `artifacts/reports/rtx5070_ctx_bench_*`
- Version extracted/tokenized/sharded outputs with the ZIM date stamp (for example `serverfault_2025-08`)
- Keep raw ZIM archives in `/mnt/ceph/llm/data/raw_zim/`
- For portable model release + offline server deploy, follow `docs/HF_RELEASE_AND_DEPLOY.md`
- On restricted-network deploy hosts, download full model snapshots locally with `hf-download-model` before serving

## Reference Material Workflow
- Store reusable project references in `information/`
- Start with `information/README.md` for curated external links
- Keep `information/raschka-reference-notes.md` updated when Raschka source material informs implementation
- Track execution progress in `information/raschka-implementation-checklist.md`
- Use `information/external/LLMs-from-scratch` (git submodule) for direct code reference
- Refresh submodule when needed with: `git submodule update --remote information/external/LLMs-from-scratch`
- When adding a new source, include a short note on why it matters to this codebase

## Wiki Maintenance
- Keep repository wiki pages source-controlled in `wiki/`
- Publish wiki updates with: `bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git`
- Keep `wiki/Dataset-Registry.md` updated with every newly approved source and intended usage
- When docs change, update `README.md`, `AGENTS.md`, and relevant `wiki/*.md` pages in the same PR
