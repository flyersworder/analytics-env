import ast
import re
import subprocess
import sys
import tempfile

import astor
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


class SQLStringFormatter(ast.NodeTransformer):
    def visit_Assign(self, node):
        # Process assignments to variables (e.g., query = "...")
        if isinstance(node.value, (ast.Str, ast.JoinedStr)):
            sql_code = ""
            if isinstance(node.value, ast.Str):
                sql_code = node.value.s
            elif isinstance(node.value, ast.JoinedStr):
                # Handle f-strings (JoinedStr)
                sql_code = "".join(
                    [
                        (
                            value.s
                            if isinstance(value, ast.Str)
                            else "{" + value.value.id + "}"
                        )
                        for value in node.value.values
                    ]
                )
            formatted_sql = format_sql_code(sql_code)
            if formatted_sql and formatted_sql.strip() != sql_code.strip():
                # Replace the original string with the formatted SQL
                if isinstance(node.value, ast.Str):
                    node.value.s = formatted_sql
                elif isinstance(node.value, ast.JoinedStr):
                    # Reconstruct the JoinedStr with formatted SQL
                    node.value.values = [ast.Str(s=formatted_sql)]
        return node


def format_sql_in_code(code):
    try:
        tree = ast.parse(code)
        formatter = SQLStringFormatter()
        formatter.visit(tree)
        formatted_code = astor.to_source(tree)
        return formatted_code
    except Exception as e:
        print(f"Error parsing code: {e}")
        return code


def format_sql_magic(code):
    lines = code.split("\n")
    new_lines = []
    in_sql_magic = False
    sql_code = ""
    for line in lines:
        if line.strip().startswith("%%sql"):
            in_sql_magic = True
            magic_line = line
            continue
        elif line.strip().startswith("%sql"):
            # Line magic; assume rest of the line is SQL code
            sql_line = line[line.find("%sql") + 4 :].strip()
            formatted_sql = format_sql_code(sql_line)
            if formatted_sql:
                new_lines.append("%sql " + formatted_sql.strip())
            else:
                new_lines.append(line)
            continue

        if in_sql_magic:
            if line.strip() == "":
                # Assume end of SQL magic
                formatted_sql = format_sql_code(sql_code)
                if formatted_sql:
                    new_lines.append(magic_line)
                    new_lines.extend(formatted_sql.strip().split("\n"))
                else:
                    new_lines.append(magic_line)
                    new_lines.extend(sql_code.strip().split("\n"))
                in_sql_magic = False
                sql_code = ""
                new_lines.append(line)  # Add the blank line
            else:
                sql_code += line + "\n"
        else:
            new_lines.append(line)
    # Handle case where SQL magic is at the end of the cell
    if in_sql_magic:
        formatted_sql = format_sql_code(sql_code)
        if formatted_sql:
            new_lines.append(magic_line)
            new_lines.extend(formatted_sql.strip().split("\n"))
        else:
            new_lines.append(magic_line)
            new_lines.extend(sql_code.strip().split("\n"))
    return "\n".join(new_lines)


def format_sql_in_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "sql" in cell.get("metadata", {}).get(
            "tags", []
        ):
            original_code = cell.source
            # First, format SQL magic commands
            code_with_formatted_magic = format_sql_magic(original_code)
            # Then, format SQL within Python strings
            formatted_code = format_sql_in_code(code_with_formatted_magic)
            if formatted_code != original_code:
                cell.source = formatted_code
                changed = True
    if changed:
        nbformat.write(nb, notebook_path)
        print(f"Formatted SQL code in {notebook_path}")


if __name__ == "__main__":
    for notebook_path in sys.argv[1:]:
        format_sql_in_notebook(notebook_path)
