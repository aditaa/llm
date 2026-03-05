PYTHONPATH=src
PYTHON=python3
ifneq ("$(wildcard .venv/bin/python)","")
PYTHON=.venv/bin/python
endif

.PHONY: setup-dev setup-train doctor install-server-system test lint format typecheck smoke extract-zim train-tokenizer sync-warm

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

sync-warm:
	@echo "Sync local extracted/shard/tokenizer artifacts to warm storage."
	@echo "Usage: bash scripts/sync_warm_storage.sh /mnt/ceph/llm/data"
