import subprocess
import sys
import tempfile

import nbformat


def format_sql_code(sql_code):
    with tempfile.NamedTemporaryFile("w+", suffix=".sql") as tmpfile:
        tmpfile.write(sql_code)
        tmpfile.flush()
        result = subprocess.run(
            ["sqlfluff", "fix", tmpfile.name, "--dialect", "ansi"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            with open(tmpfile.name, "r") as formatted_file:
                formatted_sql = formatted_file.read()
            return formatted_sql
        else:
            print(f"Error formatting SQL code:\n{result.stderr}")
            return None


def format_sql_in_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "sql" in cell.get("metadata", {}).get(
            "tags", []
        ):
            original_code = cell.source
            code_without_magic = original_code.replace("%%sql", "").strip()
            formatted_code = format_sql_code(code_without_magic)
            if formatted_code and formatted_code.strip() != code_without_magic.strip():
                # Re-add the magic command if it was present
                if "%%sql" in original_code:
                    formatted_code = "%%sql\n" + formatted_code
                cell.source = formatted_code
                changed = True
    if changed:
        nbformat.write(nb, notebook_path)
        print(f"Formatted SQL code in {notebook_path}")


if __name__ == "__main__":
    for notebook_path in sys.argv[1:]:
        format_sql_in_notebook(notebook_path)
