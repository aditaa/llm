PYTHONPATH=src
PYTHON=python3
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON=.venv/bin/python
endif

.PHONY: setup-dev setup-train setup-infer doctor install-server-system install-systemd-services test lint format typecheck smoke extract-zim train-tokenizer train-tokenizer-global corpus-quality-report clean-corpus-batch dataset-risk-report pull-hf-rows parquet-to-corpus fineweb-parquet-to-shards stage-fineweb-from-warm fineweb-prefetch-hot-queue fineweb-stage-shard-loop fineweb-hot-queue lr-sweep-350bt train-350bt-v2 train-350bt-ctx1024 train-supervisor-350bt pipeline-eta pipeline-live shard-corpus-batch verify-shards train generate average-checkpoints eval-checkpoint render-eval-dashboard package-inference-bundle sync-warm hydrate-warm offload-zim hf-download-resumable hf-download-watchdog hf-prepare-publish hf-download-model serve-openai publish-wiki

setup-dev:
	bash scripts/bootstrap_dev.sh

setup-train:
	bash scripts/bootstrap_train.sh

setup-infer:
	bash scripts/bootstrap_inference.sh

doctor:
	bash scripts/doctor.sh

install-server-system:
	bash scripts/install_server_system.sh

install-systemd-services:
	bash scripts/install_systemd_services.sh --install-watchdog

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_*.py"

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check src tests

format:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m black src tests

typecheck:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mypy src

smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m llm.cli stats --input README.md

extract-zim:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli extract-zim-text --input-zim /path/to/file.zim --output data/extracted/corpus.txt"

train-tokenizer:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train-tokenizer --input data/cleaned/corpus.clean.txt --output artifacts/tokenizer/vocab.json --bpe-vocab-size 32000"

train-tokenizer-global:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train-tokenizer-global --input-dir data/cleaned --pattern '*.clean.txt' --from-shards-path data/shards --output artifacts/tokenizer/global-bpe-v1.json --bpe-vocab-size 32000"

corpus-quality-report:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli corpus-quality-report --input-dir data/extracted --output artifacts/reports/corpus_quality.json"

clean-corpus-batch:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli clean-corpus-batch --input-dir data/extracted --output-dir data/cleaned --boilerplate-report artifacts/reports/corpus_quality.json --en-only"

dataset-risk-report:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli dataset-risk-report --input-dir data/extracted --output artifacts/reports/dataset_risk.json"

pull-hf-rows:
	@echo "Usage:"
	@echo "  python3 scripts/pull_hf_rows.py --dataset HuggingFaceFW/fineweb --config sample-350BT --split train --output /mnt/ceph/llm/data/extracted/fineweb_sample-350BT_rows100k.txt --max-rows 100000"

parquet-to-corpus:
	@echo "Usage:"
	@echo "  python3 scripts/parquet_to_corpus.py --input-dir data/fineweb/sample-350BT --output-dir data/extracted/fineweb/sample-350BT --field text"

fineweb-parquet-to-shards:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/fineweb_parquet_to_shards.py --input-dir data/fineweb/sample-350BT --output-dir data/shards_global/fineweb-global-bpe-v1 --tokenizer-out artifacts/tokenizer/fineweb-global-bpe-v1.json --bpe-vocab-size 32000 --field text"

stage-fineweb-from-warm:
	@echo "Usage:"
	@echo "  bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8"

fineweb-prefetch-hot-queue:
	@echo "Usage:"
	@echo "  bash scripts/fineweb_prefetch_hot_queue.sh --queue-min-files 12 --stage-max-files 8 --sleep-seconds 60"

fineweb-stage-shard-loop:
	@echo "Usage:"
	@echo "  bash scripts/fineweb_stage_shard_loop.sh --hot-queue-min-files 8 --stage-max-files 2 --process-max-files 4 --shard-jobs 2 --tokenizer-threads 10 --encode-batch-size 1024 --sleep-seconds 60 --shard-min-batch-size 512"

fineweb-hot-queue:
	@echo "Usage:"
	@echo "  bash scripts/fineweb_stage_shard_loop.sh --hot-queue-min-files 12 --stage-max-files 10 --process-max-files 10 --shard-jobs 2 --tokenizer-threads 10 --encode-batch-size 1024 --sleep-seconds 60"

lr-sweep-350bt:
	@echo "Usage:"
	@echo "  bash scripts/lr_sweep_rtx5070_fineweb_350bt_ctx512.sh"

train-350bt-v2:
	@echo "Usage:"
	@echo "  bash scripts/train_rtx5070_fineweb_350bt_bpe_v2.sh"

train-350bt-ctx1024:
	@echo "Usage:"
	@echo "  bash scripts/train_rtx5070_fineweb_350bt_bpe_v2_ctx1024.sh"

train-supervisor-350bt:
	@echo "Usage:"
	@echo "  bash scripts/train_supervisor_rtx5070_350bt.sh --step-chunk 2000 --poll-seconds 60 --batch-size 12 --target-effective-batch 24 --min-batch-size 6 --max-batch-size 20 --batch-step 2 --checkpoint-keep-last 6 --checkpoint-keep-every 10000 --ema-decay 0.999 --eval-suite configs/eval/standard_prompt_suite_v3.json --no-train-fail-on-eval-regression"

pipeline-eta:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/pipeline_eta_report.py --loop --interval-seconds 60"

pipeline-live:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/pipeline_live_view.py --refresh-seconds 5"
	@echo "  # live-only monitor (system + pipeline tasks), no report file writes"

shard-corpus-batch:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli shard-corpus-batch --input-dir data/cleaned --pattern '*.clean.txt' --from-shards-path data/shards --tokenizer artifacts/tokenizer/global-bpe-v1.json --output-root data/shards_global/global-bpe-v1"

verify-shards:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli verify-shards --path data/shards --raw-zim-dir data/raw_zim --strict-source"

train:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train --shards-path data/shards/<dataset> --output-dir artifacts/checkpoints/<run_name> --lr-schedule cosine --lr-warmup-steps 200 --grad-accum-steps 1 --checkpoint-keep-last 6 --checkpoint-keep-every 10000 --fail-on-eval-regression"

generate:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli generate --checkpoint artifacts/checkpoints/<run_name>/last.pt --prompt 'Hello'"

average-checkpoints:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli average-checkpoints --checkpoint artifacts/checkpoints/<run_name>/ckpt_step_0001000.pt --checkpoint artifacts/checkpoints/<run_name>/ckpt_step_0002000.pt --output artifacts/checkpoints/<run_name>/avg_last2.pt --state-key model_state"

eval-checkpoint:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/eval_checkpoint_prompts.py --checkpoint artifacts/checkpoints/<run_name>/last.pt --suite configs/eval/standard_prompt_suite_v3.json --baseline-report artifacts/reports/evals/<baseline>.json --promotion-policy configs/eval/promotion_policy_v1.json --fail-on-regression"

render-eval-dashboard:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/render_eval_trend_dashboard.py --input-tsv artifacts/reports/train_supervisor_350bt/eval_trend.tsv --output-html artifacts/reports/train_supervisor_350bt/eval_dashboard.html --output-json artifacts/reports/train_supervisor_350bt/eval_dashboard_summary.json"

package-inference-bundle:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/package_inference_bundle.py --checkpoint artifacts/checkpoints/<run_name>/best.pt --model-id local/<model_name> --create-tar"

sync-warm:
	@echo "Sync local raw/training data + artifacts to warm storage."
	@echo "Usage: bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data"

hydrate-warm:
	@echo "Hydrate local hot workspace from warm storage cache."
	@echo "Usage: bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data"

offload-zim:
	@echo "Continuously move raw ZIMs from hot to warm storage."
	@echo "Usage: bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120"

hf-download-resumable:
	@echo "Run a self-healing Hugging Face download worker with resume + retries."
	@echo "Usage: HF_TOKEN=hf_xxx bash scripts/hf_download_resumable.sh --dataset HuggingFaceFW/fineweb --repo-type dataset --include 'sample/350BT/*.parquet' --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT --max-workers 6 --enable-hf-transfer --skip-dry-run --attempt-timeout-seconds 5400 --retry-delay-seconds 30 --max-retries 0 --log-file artifacts/reports/fineweb_350bt_download_resumable.log"

hf-download-watchdog:
	@echo "Run watchdog wrapper that restarts stalled or exited HF download worker."
	@echo "Usage: HF_TOKEN=hf_xxx bash scripts/hf_download_watchdog.sh --dataset HuggingFaceFW/fineweb --repo-type dataset --include 'sample/350BT/*.parquet' --local-dir /mnt/ceph/llm/data/fineweb/sample-350BT --max-workers 4 --enable-hf-transfer --skip-dry-run --attempt-timeout-seconds 5400 --stall-seconds 1200 --worker-log-file artifacts/reports/fineweb_350bt_download_resumable.log --watchdog-log-file artifacts/reports/hf_download_watchdog.log"

hf-prepare-publish:
	@echo "Prepare release bundle and optionally push to Hugging Face model repo."
	@echo "Usage: $(PYTHON) scripts/hf_prepare_and_publish_model.py --repo-id <owner/model> --checkpoint artifacts/checkpoints/<run>/last.pt --include-safetensors [--push]"

hf-download-model:
	@echo "Download full model snapshot from Hugging Face to local directory."
	@echo "Usage: bash scripts/hf_download_model.sh <owner/model> <dest-dir>"

serve-openai:
	@echo "Serve local model as OpenAI-compatible API."
	@echo "Usage: bash scripts/run_openai_server.sh <model-dir>"

publish-wiki:
	@echo "Publish wiki pages from ./wiki to GitHub wiki repo."
	@echo "Usage: bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git"
