.PHONY: help install install-web ui demo test lint typecheck fmt grobid clean

help:
	@echo "install    Install package + dev/vector extras (uv)"
	@echo "install-web Add the web extras (FastAPI + uvicorn)"
	@echo "ui         Serve the ask-a-question UI on :8000"
	@echo "demo       Assemble the Pages demo and serve it on :8001"
	@echo "test       Run the deterministic unit tests (no API keys needed)"
	@echo "lint       Ruff lint"
	@echo "fmt        Ruff format + import sort"
	@echo "typecheck  mypy on src/rpsg"
	@echo "grobid     Start GROBID PDF-parsing service on :8070 (docker)"
	@echo "clean      Remove caches"

install:
	uv pip install -e ".[dev,vector]"

install-web:
	uv pip install -e ".[web]"

# Single-worker on purpose: the arms hold FAISS and Kuzu handles that are not shared
# across processes, and each worker would load its own copy of the embedder.
ui:
	uvicorn rpsg.web.app:app --host 127.0.0.1 --port 8000 --workers 1

# Assembled exactly as .github/workflows/pages.yml assembles it, so a local preview
# and the published site are the same three-file bundle.
demo:
	rm -rf _site && mkdir -p _site
	cp docs/demo/index.html docs/demo/recordings.json _site/
	cp src/rpsg/web/static/style.css src/rpsg/web/static/render.js _site/
	cd _site && python -m http.server 8001

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