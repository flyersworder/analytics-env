import ast
import re
import subprocess
import sys
import tempfile

import astor
import nbformat

# Determine if we're using Python 3.8 or higher
PY38_PLUS = sys.version_info >= (3, 8)


def format_sql_code(sql_code):
    with tempfile.NamedTemporaryFile("w+", suffix=".sql") as tmpfile:
        tmpfile.write(sql_code)
        tmpfile.flush()
        result = subprocess.run(
            [
                "sqlfluff",
                "fix",
                tmpfile.name,
                "--dialect",
                "ansi",
                "--templater",
                "jinja",
                "--disable-progress-bar",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode in (0, 1):
            if "templating/parsing errors" in result.stderr:
                print(
                    f"Parsing error encountered. Skipping formatting for this code block."
                )
                print("SQL code causing the error:")
                print(sql_code)
                return sql_code  # Return the original code unmodified
            else:
                with open(tmpfile.name, "r") as formatted_file:
                    formatted_sql = formatted_file.read()
                return formatted_sql
        else:
            print(f"Error formatting SQL code:\n{result.stderr}")
            return sql_code  # Return the original code unmodified


class SQLStringFormatter(ast.NodeTransformer):
    SQL_KEYWORDS = {
        "SELECT",
        "FROM",
        "WHERE",
        "INSERT",
        "UPDATE",
        "DELETE",
        "WITH",
        "JOIN",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "UNION",
        "EXCEPT",
        "INTERSECT",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
    }
    FUNCTION_NAMES = {
        "execute",
        "executescript",
        "fetchall",
        "fetchone",
        "fetchmany",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
    }

    def is_string_node(self, node):
        if PY38_PLUS:
            return isinstance(node, ast.Constant) and isinstance(node.value, str)
        else:
            return isinstance(node, ast.Str)

    def get_string_value(self, node):
        if PY38_PLUS:
            return node.value
        else:
            return node.s

    def set_string_value(self, node, new_value):
        if PY38_PLUS:
            node.value = new_value
        else:
            node.s = new_value

    def contains_sql_keywords(self, sql_code):
        upper_sql = sql_code.upper()
        # Require at least two different SQL keywords to reduce false positives
        found_keywords = [
            keyword for keyword in self.SQL_KEYWORDS if keyword in upper_sql
        ]
        return len(found_keywords) >= 2

    def visit_Assign(self, node):
        if self.is_string_node(node.value) or isinstance(node.value, ast.JoinedStr):
            sql_code, format_values = self.extract_sql_from_string(node.value)
            print(f"Found assignment with potential SQL code:\n{sql_code}")
            if self.contains_sql_keywords(sql_code):
                print("SQL keywords detected. Formatting SQL code.")
                try:
                    formatted_sql = format_sql_code(sql_code)
                    if formatted_sql and formatted_sql.strip() != sql_code.strip():
                        node.value = self.build_string_node(
                            formatted_sql, format_values
                        )
                except Exception as e:
                    print(f"Exception during formatting: {e}")
            else:
                print("No SQL keywords detected. Skipping formatting.")
        return self.generic_visit(node)

    def visit_Call(self, node):
        func_name = self.get_full_func_name(node.func)
        if any(func in func_name for func in self.FUNCTION_NAMES):
            for idx, arg in enumerate(node.args):
                if self.is_string_node(arg) or isinstance(arg, ast.JoinedStr):
                    sql_code, format_values = self.extract_sql_from_string(arg)
                    print(
                        f"Found function call '{func_name}' with potential SQL code:\n{sql_code}"
                    )
                    if self.contains_sql_keywords(sql_code):
                        print("SQL keywords detected. Formatting SQL code.")
                        try:
                            formatted_sql = format_sql_code(sql_code)
                            if (
                                formatted_sql
                                and formatted_sql.strip() != sql_code.strip()
                            ):
                                node.args[idx] = self.build_string_node(
                                    formatted_sql, format_values
                                )
                        except Exception as e:
                            print(f"Exception during formatting: {e}")
                    else:
                        print("No SQL keywords detected. Skipping formatting.")
        return self.generic_visit(node)

    def get_full_func_name(self, node):
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            names.append(node.id)
        return ".".join(reversed(names))

    def extract_sql_from_string(self, node):
        sql_code = ""
        format_values = []
        placeholder_name = "__SQL_PLACEHOLDER__"  # Use a unique placeholder
        if self.is_string_node(node):
            sql_code = self.get_string_value(node)
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                if self.is_string_node(value):
                    sql_code += self.get_string_value(value)
                elif isinstance(value, ast.FormattedValue):
                    sql_code += placeholder_name
                    format_values.append(value)
        return sql_code, format_values

    def build_string_node(self, formatted_sql, format_values):
        if not format_values:
            if PY38_PLUS:
                return ast.Constant(value=formatted_sql)
            else:
                return ast.Str(s=formatted_sql)
        else:
            placeholder_pattern = re.compile(re.escape("__SQL_PLACEHOLDER__"))
            sql_parts = placeholder_pattern.split(formatted_sql)
            new_values = []
            num_placeholders = len(format_values)

            # Reconstruct the f-string
            for idx, part in enumerate(sql_parts):
                if part:
                    if PY38_PLUS:
                        new_values.append(ast.Constant(value=part))
                    else:
                        new_values.append(ast.Str(s=part))
                if idx < num_placeholders:
                    new_values.append(format_values[idx])
            return ast.JoinedStr(values=new_values)


def format_sql_magic(code):
    lines = code.split("\n")
    new_lines = []
    in_sql_magic = False
    sql_code = ""
    magic_line = ""
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("%%sql"):
            in_sql_magic = True
            magic_line = line
            sql_code = ""
            continue
        elif stripped_line.startswith("%sql"):
            # Line magic; assume rest of the line is SQL code
            sql_line = line[line.find("%sql") + 4 :].strip()
            if sql_line:
                if contains_sql_keywords(sql_line):
                    formatted_sql = format_sql_code(sql_line)
                    if formatted_sql:
                        new_lines.append("%sql " + formatted_sql.strip())
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            continue

        if in_sql_magic:
            if stripped_line == "":
                # End of SQL magic block
                if contains_sql_keywords(sql_code):
                    formatted_sql = format_sql_code(sql_code)
                    if formatted_sql:
                        new_lines.append(magic_line)
                        new_lines.extend(formatted_sql.strip().split("\n"))
                    else:
                        new_lines.append(magic_line)
                        new_lines.extend(sql_code.strip().split("\n"))
                else:
                    new_lines.append(magic_line)
                    new_lines.extend(sql_code.strip().split("\n"))
                in_sql_magic = False
                new_lines.append(line)  # Add the blank line
            else:
                sql_code += line + "\n"
        else:
            new_lines.append(line)
    # Handle case where SQL magic is at the end of the cell
    if in_sql_magic:
        if contains_sql_keywords(sql_code):
            formatted_sql = format_sql_code(sql_code)
            if formatted_sql:
                new_lines.append(magic_line)
                new_lines.extend(formatted_sql.strip().split("\n"))
            else:
                new_lines.append(magic_line)
                new_lines.extend(sql_code.strip().split("\n"))
        else:
            new_lines.append(magic_line)
            new_lines.extend(sql_code.strip().split("\n"))
    return "\n".join(new_lines)


def contains_sql_keywords(sql_code):
    SQL_KEYWORDS = {
        "SELECT",
        "FROM",
        "WHERE",
        "INSERT",
        "UPDATE",
        "DELETE",
        "WITH",
        "JOIN",
        "CREATE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "UNION",
        "EXCEPT",
        "INTERSECT",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
    }
    upper_sql = sql_code.upper()
    found_keywords = [keyword for keyword in SQL_KEYWORDS if keyword in upper_sql]
    return len(found_keywords) >= 2  # Require at least two keywords


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


def format_sql_in_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    changed = False
    for cell in nb.cells:
        # Optionally, reintroduce the 'sql' tag requirement
        # if cell.cell_type == 'code' and 'sql' in cell.get('metadata', {}).get('tags', []):
        if cell.cell_type == "code":
            original_code = cell.source
            # First, format SQL magic commands
            code_with_formatted_magic = format_sql_magic(original_code)
            # Then, format SQL within Python code
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
