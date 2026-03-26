.PHONY: setup lint format test docs clean

setup:
	uv sync --all-extras
	prek install

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run ruff check --fix .
	uv run black .

test:
	uv run pytest -v

docs:
	cd notebooks && uv run quarto render

clean:
	rm -rf .ruff_cache .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
