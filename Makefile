PYTHONPATH=src
PYTHON=python3

.PHONY: setup-dev setup-train doctor install-server-system test lint format typecheck smoke

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
