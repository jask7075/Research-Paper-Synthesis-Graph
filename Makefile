.PHONY: help install test lint typecheck fmt grobid clean

help:
	@echo "install    Install package + dev/vector extras (uv)"
	@echo "test       Run the deterministic unit tests (no API keys needed)"
	@echo "lint       Ruff lint"
	@echo "fmt        Ruff format + import sort"
	@echo "typecheck  mypy on src/rpsg"
	@echo "grobid     Start GROBID PDF-parsing service on :8070 (docker)"
	@echo "clean      Remove caches"

install:
	uv pip install -e ".[dev,vector]"

test:
	pytest

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck:
	mypy

grobid:
	docker run --rm -t --init -p 8070:8070 lfoppiano/grobid:0.8.0

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +