PYTHONPATH=src
PYTHON=python3
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON=.venv/bin/python
endif

.PHONY: setup-dev setup-train doctor install-server-system test lint format typecheck smoke extract-zim train-tokenizer verify-shards train sync-warm hydrate-warm publish-wiki

setup-dev:
	bash scripts/bootstrap_dev.sh

setup-train:
	bash scripts/bootstrap_train.sh

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

verify-shards:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli verify-shards --path data/shards --raw-zim-dir data/raw_zim --strict-source"

train:
	@echo "Usage:"
	@echo "  PYTHONPATH=src $(PYTHON) -m llm.cli train --shards-path data/shards/<dataset> --output-dir artifacts/checkpoints/<run_name>"

sync-warm:
	@echo "Sync local extracted/shard/tokenizer artifacts to warm storage."
	@echo "Usage: bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data"

hydrate-warm:
	@echo "Hydrate local hot workspace from warm storage cache."
	@echo "Usage: bash scripts/hydrate_from_warm_storage.sh /mnt/ceph/llm/data"

publish-wiki:
	@echo "Publish wiki pages from ./wiki to GitHub wiki repo."
	@echo "Usage: bash scripts/publish_wiki.sh git@github.com:aditaa/llm.wiki.git"
