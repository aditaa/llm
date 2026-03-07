PYTHONPATH=src
PYTHON=python3
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON=.venv/bin/python
endif

.PHONY: setup-dev setup-train setup-infer doctor install-server-system test lint format typecheck smoke extract-zim train-tokenizer train-tokenizer-global corpus-quality-report clean-corpus-batch dataset-risk-report pull-hf-rows parquet-to-corpus fineweb-parquet-to-shards stage-fineweb-from-warm fineweb-stage-shard-loop shard-corpus-batch verify-shards train generate eval-checkpoint sync-warm hydrate-warm offload-zim hf-prepare-publish hf-download-model serve-openai publish-wiki

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
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train-tokenizer --input data/extracted/corpus.txt --output artifacts/tokenizer/vocab.json"

train-tokenizer-global:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train-tokenizer-global --input-dir data/extracted --from-shards-path data/shards --output artifacts/tokenizer/global-char-v1.json"

corpus-quality-report:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli corpus-quality-report --input-dir data/extracted --output artifacts/reports/corpus_quality.json"

clean-corpus-batch:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli clean-corpus-batch --input-dir data/extracted --output-dir data/cleaned --boilerplate-report artifacts/reports/corpus_quality.json"

dataset-risk-report:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli dataset-risk-report --input-dir data/extracted --output artifacts/reports/dataset_risk.json"

pull-hf-rows:
	@echo "Usage:"
	@echo "  python3 scripts/pull_hf_rows.py --dataset HuggingFaceFW/fineweb --config sample-10BT --split train --output /mnt/ceph/llm/data/extracted/fineweb_sample-10BT_rows100k.txt --max-rows 100000"

parquet-to-corpus:
	@echo "Usage:"
	@echo "  python3 scripts/parquet_to_corpus.py --input-dir data/fineweb/sample-10BT --output-dir data/extracted/fineweb/sample-10BT --field text"

fineweb-parquet-to-shards:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/fineweb_parquet_to_shards.py --input-dir data/fineweb/sample-10BT --output-dir data/shards_global/fineweb-s10bt-global-char-v1 --tokenizer-out artifacts/tokenizer/fineweb-s10bt-global-char-v1.json --field text"

stage-fineweb-from-warm:
	@echo "Usage:"
	@echo "  bash scripts/stage_fineweb_from_warm.sh --max-files 4 --max-gib 8"

fineweb-stage-shard-loop:
	@echo "Usage:"
	@echo "  bash scripts/fineweb_stage_shard_loop.sh --stage-max-files 10 --process-max-files 10 --sleep-seconds 120"

shard-corpus-batch:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli shard-corpus-batch --input-dir data/extracted --from-shards-path data/shards --tokenizer artifacts/tokenizer/global-char-v1.json --output-root data/shards_global/global-char-v1"

verify-shards:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli verify-shards --path data/shards --raw-zim-dir data/raw_zim --strict-source"

train:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train --shards-path data/shards/<dataset> --output-dir artifacts/checkpoints/<run_name>"

generate:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli generate --checkpoint artifacts/checkpoints/<run_name>/last.pt --prompt 'Hello'"

eval-checkpoint:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) scripts/eval_checkpoint_prompts.py --checkpoint artifacts/checkpoints/<run_name>/last.pt --suite configs/eval/standard_prompt_suite_v1.json"

sync-warm:
	@echo "Sync local raw/training data + artifacts to warm storage."
	@echo "Usage: bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data"

hydrate-warm:
	@echo "Hydrate local hot workspace from warm storage cache."
	@echo "Usage: bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data"

offload-zim:
	@echo "Continuously move raw ZIMs from hot to warm storage."
	@echo "Usage: bash scripts/zim_offload_worker.sh data/raw_zim /mnt/ceph/llm/data/raw_zim 120"

hf-prepare-publish:
	@echo "Prepare release bundle and optionally push to Hugging Face model repo."
	@echo "Usage: $(PYTHON) scripts/hf_prepare_and_publish_model.py --repo-id <owner/model> --checkpoint artifacts/checkpoints/<run>/last.pt [--push]"

hf-download-model:
	@echo "Download full model snapshot from Hugging Face to local directory."
	@echo "Usage: bash scripts/hf_download_model.sh <owner/model> <dest-dir>"

serve-openai:
	@echo "Serve local model as OpenAI-compatible API."
	@echo "Usage: bash scripts/run_openai_server.sh <model-dir>"

publish-wiki:
	@echo "Publish wiki pages from ./wiki to GitHub wiki repo."
	@echo "Usage: bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git"
