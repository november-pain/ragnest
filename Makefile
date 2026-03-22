.PHONY: install lint format typecheck test test-unit test-integration

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/
	basedpyright src/

test:
	pytest

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/ -m integration
