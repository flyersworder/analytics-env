# Analytics Template Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform analytics-env into a clean, reusable team template with security fixes, modular dependencies, prek hooks, testing, CI, and documentation.

**Architecture:** Six sequential tasks — each produces a self-contained, committable change. Security cleanup first (untrack secrets), then dependency restructuring, prek migration, test infrastructure, CI pipeline, and finally Makefile + README.

**Tech Stack:** Python 3.12, uv, prek, pytest, GitHub Actions, Quarto

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `.env.example` | Placeholder API keys for onboarding |
| Modify | `.gitignore` | Already has `.env` — verify only |
| Modify | `pyproject.toml` | Description, dependency groups, pytest config |
| Delete | `docs/Interview_Preparation_FAQ.md` | Personal doc, not template material |
| Delete | `docs/Coding_Interview_Pre_FAQ.md` | Personal doc, not template material |
| Delete | `docs/Coding_FAQ_sup.md` | Personal doc, not template material |
| Delete | `tests/test_sql.ipynb` | Scratchpad, not real tests |
| Create | `tests/conftest.py` | Pytest fixture placeholder |
| Create | `tests/test_notebooks.py` | Notebook parse validation |
| Create | `.github/workflows/ci.yml` | Lint + test CI pipeline |
| Create | `Makefile` | Task runner for common commands |
| Create | `README.md` | Team onboarding documentation (overwrite empty file) |

---

### Task 1: Security & Cleanup

**Files:**
- Create: `.env.example`
- Modify: `.gitignore` (verify `.env` entry exists — it does at line 131)
- Delete: `docs/Interview_Preparation_FAQ.md`
- Delete: `docs/Coding_Interview_Pre_FAQ.md`
- Delete: `docs/Coding_FAQ_sup.md`
- Delete: `tests/test_sql.ipynb`

- [ ] **Step 1: Untrack `.env` from git**

```bash
git rm --cached .env
```

Expected: `rm '.env'` — file stays on disk, removed from index.

- [ ] **Step 2: Create `.env.example`**

Create `.env.example` with placeholder values:

```
GOOGLE_API_KEY=your-google-api-key
GOOGLE_MAPS_API_KEY=your-google-maps-api-key
PATENT_SEARCH_API_KEY=your-patent-search-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

- [ ] **Step 3: Delete interview docs and test scratchpad**

```bash
git rm docs/Interview_Preparation_FAQ.md
git rm docs/Coding_Interview_Pre_FAQ.md
git rm docs/Coding_FAQ_sup.md
git rm tests/test_sql.ipynb
```

- [ ] **Step 4: Commit**

```bash
git add .env.example
git commit -m "Remove tracked secrets and personal docs

- Untrack .env (was committed with real API keys)
- Add .env.example with placeholder values
- Remove interview prep docs (not template material)
- Remove test_sql.ipynb scratchpad"
```

---

### Task 2: Restructure Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Rewrite pyproject.toml dependencies section**

Replace the flat `dependencies` list and add optional dependency groups. Update the project description.

```toml
[project]
name = "analytics-env"
version = "0.1.0"
description = "A standardized analytics environment template with pre-configured tooling for data science, LLM integration, and document processing"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "duckdb>=1.2.1",
    "getdaft>=0.3.13",
    "ipywidgets>=8.1.5",
    "jupysql>=0.10.14",
    "kaleido==0.2.1",
    "langextract>=1.0.8",
    "marimo>=0.11.30",
    "matplotlib>=3.9.2",
    "nbformat>=5.10.4",
    "notebook>=7.2.2",
    "pandas>=2.2.3",
    "plotly>=5.24.1",
    "python-dotenv>=1.0.1",
    "requests>=2.32.3",
    "scipy>=1.14.1",
    "seaborn>=0.13.2",
    "statsmodels>=0.14.4",
    "tabulate>=0.9.0",
    "thefuzz>=0.22.1",
    "tqdm>=4.67.1",
]

[project.optional-dependencies]
llm = [
    "aisuite[all]>=0.1.6",
    "crewai>=0.86.0",
    "langchain>=0.3.13",
    "langchain-experimental>=0.3.4",
    "langchain-google-genai>=2.0.7",
    "langchain-google-vertexai>=2.0.9",
    "llama-index>=0.12.1",
    "tavily-python>=0.5.0",
]
pdf = [
    "docling>=2.37.0",
    "pdf2docx>=0.5.8",
    "pdfplumber>=0.11.7",
    "pymupdf4llm>=0.0.17",
    "reportlab>=4.3.1",
]
dev = [
    "astor>=0.8.1",
    "black>=24.10.0",
    "nbqa>=1.9.1",
    "pytest>=8.4.2",
    "sqlfluff>=3.2.4",
    "sqlglot[rs]>=25.25.1",
    "sqlnbfmt>=0.1.0",
    "sqlparse>=0.5.1",
]
all = [
    "analytics-env[llm]",
    "analytics-env[pdf]",
    "analytics-env[dev]",
]
```

Note: `docker>=7.1.0` is removed — it was unused as a Python dependency (Docker CLI is a system tool).

- [ ] **Step 2: Add pytest configuration**

Append to `pyproject.toml` after the existing `[tool.ruff]` section:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Run `uv sync --all-extras` to verify resolution**

```bash
uv sync --all-extras
```

Expected: resolves and installs without errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Restructure dependencies into core/llm/pdf/dev groups

- Core: data science essentials (pandas, plotly, duckdb, etc.)
- llm: LangChain, CrewAI, LlamaIndex, etc.
- pdf: docling, pdfplumber, pymupdf4llm, etc.
- dev: linting, formatting, testing tools
- all: shorthand for all extras
- Add pytest config
- Remove docker dep (unused as Python package)
- Update project description"
```

---

### Task 3: Replace pre-commit with prek

**Files:**
- Modify: `pyproject.toml` (remove `pre-commit` from deps — already done in Task 2)

No file changes needed — `.pre-commit-config.yaml` stays as-is. This task is about installing prek and verifying compatibility.

- [ ] **Step 1: Install prek**

```bash
brew install prek
```

Or if brew is not available:

```bash
uv tool install prek
```

- [ ] **Step 2: Install git hooks via prek**

```bash
prek install -f
```

Expected: hooks installed in `.git/hooks/pre-commit`. The `-f` flag overwrites the existing pre-commit hook.

- [ ] **Step 3: Verify hooks work**

```bash
prek run --all-files
```

Expected: all hooks run and pass (ruff, black, sqlnbfmt, nbqa-black, nbqa-ruff, sqlfluff-fix, nbstripout, end-of-file-fixer, trailing-whitespace, check-added-large-files, check-ast).

- [ ] **Step 4: Commit** (this commit itself validates that prek hooks fire on commit)

```bash
git commit --allow-empty -m "Switch from pre-commit to prek for git hooks

prek is a faster, Rust-based drop-in replacement. Config file
(.pre-commit-config.yaml) unchanged — remains compatible with
pre-commit if needed."
```

---

### Task 4: Test Infrastructure

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_notebooks.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Shared test fixtures for analytics-env."""
```

- [ ] **Step 2: Create `tests/test_notebooks.py`**

```python
"""Validate that all Jupyter notebooks in notebooks/ are well-formed."""

from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path("notebooks")


def get_notebook_paths():
    """Collect all .ipynb files in the notebooks directory."""
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook_path",
    get_notebook_paths(),
    ids=lambda p: p.name,
)
def test_notebook_parses(notebook_path):
    """Each notebook should be valid nbformat v4."""
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    assert nb.cells, f"{notebook_path.name} has no cells"
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
uv run pytest tests/ -v
```

Expected: 4 tests pass (one per notebook: GoogleGenAI.ipynb, langextract.ipynb, patent_analysis.ipynb, pdf2docx.ipynb).

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/test_notebooks.py
git commit -m "Add pytest infrastructure with notebook validation tests

Parametrized test validates all notebooks in notebooks/ parse
as valid nbformat v4 and contain at least one cell."
```

---

### Task 5: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: j178/prek-action@v1

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv sync --extra dev
      - run: uv run pytest -v
```

- [ ] **Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "Add GitHub Actions CI with prek lint and pytest

Two jobs: lint (prek-action runs all hooks) and test (pytest
with dev dependencies)."
```

---

### Task 6: Makefile & README

**Files:**
- Create: `Makefile`
- Create: `README.md` (overwrite empty file)

- [ ] **Step 1: Create `Makefile`**

```makefile
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
```

- [ ] **Step 2: Create `README.md`**

```markdown
# analytics-env

A standardized analytics environment template with pre-configured tooling for data science, LLM integration, and document processing.

## Features

- **Dependency management** with [uv](https://docs.astral.sh/uv/) and modular dependency groups (core, llm, pdf, dev)
- **Code quality** via [prek](https://github.com/j178/prek) git hooks — ruff, black, sqlfluff, nbqa, nbstripout
- **SQL formatting** in Jupyter notebooks with sqlnbfmt
- **Testing** with pytest and automated notebook validation
- **CI/CD** with GitHub Actions (lint + test on every PR)
- **Notebook rendering** with Quarto (HTML and PDF with Plotly support)

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [prek](https://github.com/j178/prek) — `brew install prek` or `uv tool install prek`

### Setup

```bash
git clone https://github.com/flyersworder/analytics-env.git
cd analytics-env
make setup
```

This installs all dependencies and configures git hooks.

### Verify

```bash
make lint && make test
```

## Project Structure

```
analytics-env/
├── notebooks/          # Jupyter notebooks and Quarto documents
│   └── _quarto.yml     # Quarto rendering config (HTML + PDF)
├── scripts/            # Standalone Python scripts
├── docs/               # Documentation and presentations
├── tests/              # pytest test suite
├── models/             # Trained model storage
├── pyproject.toml      # Project config, dependencies, tool settings
├── .pre-commit-config.yaml  # Hook definitions (used by prek)
├── .sqlfluff           # SQL linting rules
├── config.yaml         # SQL formatting config for sqlnbfmt
├── Makefile            # Common task runner
└── .env.example        # Required environment variables (copy to .env)
```

## Dependency Groups

Install only what you need:

| Group | Install command | What's included |
|-------|----------------|-----------------|
| **Core** | `uv sync` | pandas, scipy, plotly, duckdb, matplotlib, seaborn, statsmodels, notebook, marimo |
| **LLM** | `uv sync --extra llm` | LangChain, CrewAI, LlamaIndex, aisuite, instructor, Tavily |
| **PDF** | `uv sync --extra pdf` | docling, pdfplumber, pymupdf4llm, pdf2docx, reportlab |
| **Dev** | `uv sync --extra dev` | black, ruff, pytest, sqlfluff, sqlglot, nbqa, nbstripout |
| **All** | `uv sync --all-extras` | Everything above |

## Development Workflow

### Environment variables

```bash
cp .env.example .env
# Fill in your API keys
```

### Git hooks

Hooks run automatically on commit via prek. To run manually:

```bash
prek run --all-files
```

### Linting and formatting

```bash
make lint     # Check style (ruff + black)
make format   # Auto-fix style
```

### Testing

```bash
make test
```

### Rendering notebooks

```bash
make docs     # Render via Quarto
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make setup` | Install all deps + git hooks |
| `make lint` | Check code style |
| `make format` | Auto-fix code style |
| `make test` | Run pytest |
| `make docs` | Render Quarto notebooks |
| `make clean` | Remove caches and build artifacts |

## Further Reading

See [docs/analytics_environment.qmd](docs/analytics_environment.qmd) for a presentation on the rationale behind this template's design decisions.
```

- [ ] **Step 3: Verify Makefile works**

```bash
make lint
make test
```

Expected: both pass.

- [ ] **Step 4: Commit**

```bash
git add Makefile README.md
git commit -m "Add Makefile and README for team onboarding

Makefile provides setup/lint/format/test/docs/clean targets.
README covers quick start, project structure, dependency groups,
and development workflow."
```

---

## Post-Implementation

After all 6 tasks are committed, run a final validation:

```bash
prek run --all-files
make test
```

Both should pass cleanly.
