# analytics-env

[![CI](https://github.com/flyersworder/analytics-env/actions/workflows/ci.yml/badge.svg)](https://github.com/flyersworder/analytics-env/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![prek](https://img.shields.io/badge/pre--commit-prek-brightgreen?logo=pre-commit)](https://github.com/j178/prek)
[![Template](https://img.shields.io/badge/use%20this-template-blue?logo=github)](https://github.com/flyersworder/analytics-env/generate)

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
| **LLM** | `uv sync --extra llm` | LangChain, CrewAI, LlamaIndex, aisuite, Tavily |
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
