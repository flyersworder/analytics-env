# Analytics Template Improvements — Design Spec

**Date:** 2026-03-26
**Status:** Approved

## Goal

Transform `analytics-env` from a working personal analytics repo into a clean, reusable team template with proper security, testing, CI, documentation, and modular dependencies.

## 1. Security & Cleanup

### Remove `.env` from tracking
- `git rm --cached .env` to untrack without deleting locally
- Verify `.env` is in `.gitignore` (add if missing)
- Create `.env.example` with placeholder values:
  ```
  GOOGLE_API_KEY=your-google-api-key
  GOOGLE_MAPS_API_KEY=your-google-maps-api-key
  PATENT_SEARCH_API_KEY=your-patent-search-api-key
  OPENROUTER_API_KEY=your-openrouter-api-key
  ```

### Remove interview prep docs
Delete from `docs/`:
- `Interview_Preparation_FAQ.md`
- `Coding_Interview_Pre_FAQ.md`
- `Coding_FAQ_sup.md`

### Fix project metadata
Update `pyproject.toml` description from "Add your description here" to a meaningful one-liner.

## 2. Dependency Groups

Restructure `pyproject.toml` from a flat dependency list into groups:

| Group | Install command | Contents |
|-------|----------------|----------|
| **core** | `uv sync` | pandas, scipy, statsmodels, matplotlib, seaborn, plotly, duckdb, notebook, marimo, ipywidgets, jupysql, tqdm, tabulate, python-dotenv, requests, thefuzz, kaleido, nbformat, getdaft, langextract |
| **llm** | `uv sync --extra llm` | langchain, langchain-google-vertexai, langchain-google-genai, langchain-experimental, crewai, llama-index, llama-index-readers-file, aisuite[all], instructor, tavily-python, cohere, cerebras-cloud-sdk, banks |
| **pdf** | `uv sync --extra pdf` | pymupdf4llm, pdfplumber, docling, pdf2docx, reportlab |
| **dev** | `uv sync --extra dev` | black, pre-commit, nbqa, pytest, astor, nbstripout, diff-cover, sqlfluff, sqlparse, sqlglot[rs], sqlnbfmt |
| **all** | `uv sync --extra all` | llm + pdf + dev |

## 3. Testing & CI

### pytest configuration
- Add `[tool.pytest.ini_options]` to `pyproject.toml`:
  - `testpaths = ["tests"]`
  - `pythonpath = ["."]`
- Create `tests/conftest.py` (empty placeholder)
- Create `tests/test_notebooks.py` — validates all `.ipynb` files in `notebooks/` parse via `nbformat.read()`
- Assess `tests/test_sql.ipynb` — convert to pytest if useful, otherwise remove

### GitHub Actions CI (`.github/workflows/ci.yml`)
- **Triggers:** push to `main`, pull requests
- **Environment:** Python 3.12, `uv` for dependency management
- **Steps:**
  1. Checkout + setup Python + install uv
  2. `uv sync --extra dev` (core + dev deps only — no API keys needed)
  3. Lint: `ruff check .`, `black --check .`
  4. Test: `pytest`
- Notebook validation happens via pytest (the `test_notebooks.py` file)

## 4. Makefile

Targets using `uv run` to stay in the managed environment:

| Target | Command | Purpose |
|--------|---------|---------|
| `setup` | `uv sync --all-extras && uv run pre-commit install` | Full install + hook setup |
| `lint` | `uv run ruff check . && uv run black --check .` | Check code style |
| `format` | `uv run ruff check --fix . && uv run black .` | Auto-fix style |
| `test` | `uv run pytest` | Run test suite |
| `docs` | `cd notebooks && uv run quarto render` | Render Quarto notebooks |
| `clean` | Remove `.ruff_cache`, `__pycache__`, `.pytest_cache`, `.quarto` | Clean build artifacts |

## 5. README

Target audience: team members onboarding to the analytics template.

Structure:
1. **Header** — project name + one-line description
2. **Features** — bullet list: toolchain, dependency groups, CI, pre-commit, Quarto
3. **Quick Start** — clone, `make setup`, verify with `make lint && make test`
4. **Project Structure** — annotated directory tree
5. **Dependency Groups** — table with group names, contents, install commands
6. **Development Workflow** — pre-commit hooks, linting, testing, notebook conventions
7. **Further Reading** — link to `docs/analytics_environment.qmd` presentation

## Out of Scope

- Docker/devcontainer setup
- Cookiecutter/copier templating
- DVC/data management configuration
- Notebook execution in CI (requires API keys)
