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
