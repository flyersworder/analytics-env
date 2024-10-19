import ast
import difflib
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import astor
import nbformat
import yaml

# Determine if we're using Python 3.8 or higher
PY38_PLUS = sys.version_info >= (3, 8)

# Set up logging
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    config_file = Path(__file__).parent / config_path
    if not config_file.is_file():
        logger.error(f"Configuration file '{config_file}' not found.")
        sys.exit(1)
    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
            sql_keywords = set(config.get("sql_keywords", []))
            function_names = set(config.get("function_names", []))
            return sql_keywords, function_names
    except yaml.YAMLError as e:
        logger.error(f"Error parsing the configuration file: {e}")
        sys.exit(1)


def format_sql_code(
    sql_code,
    sqlfluff_config_path,
    dialect,
    sqlfluff_rules="",
    sqlfluff_fix_args="",
):
    """Formats the given SQL code using sqlfluff."""
    with tempfile.NamedTemporaryFile("w+", suffix=".sql", delete=False) as tmpfile:
        tmpfile.write(sql_code)
        tmpfile.flush()
        tmpfile_path = tmpfile.name

    try:
        sqlfluff_command = [
            "sqlfluff",
            "fix",
            tmpfile_path,
            "--dialect",
            dialect,
            "--templater",
            "jinja",
            "--disable-progress-bar",
        ]

        if sqlfluff_rules:
            sqlfluff_command.extend(["--rules", sqlfluff_rules])

        if sqlfluff_fix_args:
            sqlfluff_command.extend(sqlfluff_fix_args.split())

        if sqlfluff_config_path:
            sqlfluff_command.extend(["--config", str(sqlfluff_config_path)])

        logger.debug(f"Running sqlfluff command: {' '.join(sqlfluff_command)}")

        result = subprocess.run(
            sqlfluff_command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode in (0, 1):
            if "templating/parsing errors" in result.stderr:
                logger.warning(
                    "Parsing error encountered. Skipping formatting for this SQL code block.",
                )
                logger.debug(f"SQL code causing the error:\n{sql_code}")
                logger.debug(f"sqlfluff stderr:\n{result.stderr}")
                return sql_code  # Return the original code unmodified
            with open(tmpfile_path) as formatted_file:
                formatted_sql = formatted_file.read()
            return formatted_sql
        logger.error(f"Error formatting SQL code:\n{result.stderr}")
        return sql_code  # Return the original code unmodified
    finally:
        # Clean up the temporary file
        Path(tmpfile_path).unlink()


class SQLStringFormatter(ast.NodeTransformer):
    """AST NodeTransformer that formats SQL strings in Assign and Call nodes."""

    def __init__(
        self,
        sql_keywords,
        function_names,
        sqlfluff_config_path,
        dialect,
        sqlfluff_rules="",
        sqlfluff_fix_args="",
    ):
        super().__init__()
        self.sql_keywords = sql_keywords
        self.function_names = function_names
        self.sqlfluff_config_path = sqlfluff_config_path
        self.dialect = dialect
        self.sqlfluff_rules = sqlfluff_rules
        self.sqlfluff_fix_args = sqlfluff_fix_args
        self.changed = False  # Track whether changes were made

    def is_string_node(self, node):
        """Checks if the node is a string node."""
        if PY38_PLUS:
            return isinstance(node, ast.Constant) and isinstance(node.value, str)
        return isinstance(node, ast.Str)

    def get_string_value(self, node):
        """Retrieves the string value from the node."""
        if PY38_PLUS:
            return node.value
        return node.s

    def set_string_value(self, node, new_value):
        """Sets the string value in the node."""
        if PY38_PLUS:
            node.value = new_value
        else:
            node.s = new_value

    def contains_sql_keywords(self, sql_code):
        """Determines if the given code contains at least two SQL keywords."""
        upper_sql = sql_code.upper()
        # Exclude strings that start with common file path indicators or URLs
        if re.match(r"^\s*(/|[a-zA-Z]:\\|https?://)", sql_code.strip()):
            return False
        # Use word boundaries to match whole words
        found_keywords = [
            keyword
            for keyword in self.sql_keywords
            if re.search(r"\b" + re.escape(keyword) + r"\b", upper_sql)
        ]
        return len(found_keywords) >= 2  # Require at least two keywords

    def visit_Assign(self, node):
        """Visits Assign nodes to find and format SQL strings."""
        if self.is_string_node(node.value) or isinstance(node.value, ast.JoinedStr):
            sql_code, format_values = self.extract_sql_from_string(node.value)
            if not sql_code:
                return self.generic_visit(node)
            logger.debug(f"Examining potential SQL code in assignment:\n{sql_code}")
            if self.contains_sql_keywords(sql_code):
                logger.info(f"SQL keywords detected in assignment:\n{sql_code}")
                try:
                    formatted_sql = format_sql_code(
                        sql_code,
                        self.sqlfluff_config_path,
                        self.dialect,
                        self.sqlfluff_rules,
                        self.sqlfluff_fix_args,
                    )
                    if formatted_sql and formatted_sql.strip() != sql_code.strip():
                        logger.info("SQL code was formatted. Showing diff:")
                        diff = difflib.unified_diff(
                            sql_code.splitlines(),
                            formatted_sql.splitlines(),
                            fromfile="before.sql",
                            tofile="after.sql",
                            lineterm="",
                        )
                        diff_text = "\n".join(diff)
                        if diff_text:
                            logger.info(diff_text)
                        # Rebuild the string node with formatted SQL
                        new_value = self.build_string_node(formatted_sql, format_values)
                        if new_value is not None:
                            node.value = new_value
                            self.changed = True  # Indicate that a change was made
                except Exception as e:
                    logger.error(f"Exception during SQL formatting: {e}")
            else:
                logger.debug(
                    "No SQL keywords detected in assignment. Skipping formatting.",
                )
        return self.generic_visit(node)

    def visit_Call(self, node):
        """Visits Call nodes to find and format SQL strings in function arguments."""
        func_name = self.get_full_func_name(node.func)
        if any(func in func_name for func in self.function_names):
            for idx, arg in enumerate(node.args):
                if self.is_string_node(arg) or isinstance(arg, ast.JoinedStr):
                    sql_code, format_values = self.extract_sql_from_string(arg)
                    if not sql_code:
                        continue
                    logger.debug(
                        f"Examining potential SQL code in function call '{func_name}':\n{sql_code}",
                    )
                    if self.contains_sql_keywords(sql_code):
                        logger.info(
                            f"SQL keywords detected in function call '{func_name}'. Formatting SQL code.",
                        )
                        try:
                            formatted_sql = format_sql_code(
                                sql_code,
                                self.sqlfluff_config_path,
                                self.dialect,
                                self.sqlfluff_rules,
                                self.sqlfluff_fix_args,
                            )
                            if (
                                formatted_sql
                                and formatted_sql.strip() != sql_code.strip()
                            ):
                                logger.info("SQL code was formatted. Showing diff:")
                                diff = difflib.unified_diff(
                                    sql_code.splitlines(),
                                    formatted_sql.splitlines(),
                                    fromfile="before.sql",
                                    tofile="after.sql",
                                    lineterm="",
                                )
                                diff_text = "\n".join(diff)
                                if diff_text:
                                    logger.info(diff_text)
                                # Rebuild the string node with formatted SQL
                                new_arg = self.build_string_node(
                                    formatted_sql,
                                    format_values,
                                )
                                if new_arg is not None:
                                    node.args[idx] = new_arg
                                    self.changed = (
                                        True  # Indicate that a change was made
                                    )
                        except Exception as e:
                            logger.error(f"Exception during SQL formatting: {e}")
                    else:
                        logger.debug(
                            "No SQL keywords detected in function call. Skipping formatting.",
                        )
        return self.generic_visit(node)

    def get_full_func_name(self, node):
        """Retrieves the full function name from the AST node."""
        names = []
        while isinstance(node, ast.Attribute):
            names.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            names.append(node.id)
        return ".".join(reversed(names))

    def extract_sql_from_string(self, node):
        """Extracts SQL code from a string node, handling f-strings."""
        sql_code = ""
        format_values = []
        placeholder_name = "__SQL_PLACEHOLDER__"  # Unique placeholder for f-strings
        if self.is_string_node(node):
            sql_code = self.get_string_value(node)
            # Allow single-line SQL if it contains multiple keywords
            if "\n" not in sql_code and not self.contains_sql_keywords(sql_code):
                return "", []
        elif isinstance(node, ast.JoinedStr):
            for value in node.values:
                if self.is_string_node(value):
                    sql_code += self.get_string_value(value)
                elif isinstance(value, ast.FormattedValue):
                    sql_code += placeholder_name
                    format_values.append(value)
            # Allow single-line SQL if it contains multiple keywords
            if "\n" not in sql_code and not self.contains_sql_keywords(sql_code):
                return "", []
        return sql_code, format_values

    def build_string_node(self, formatted_sql, format_values):
        """Reconstructs the string node with formatted SQL."""
        if not format_values:
            if PY38_PLUS:
                return ast.Constant(value=formatted_sql)
            return ast.Str(s=formatted_sql)
        placeholder_pattern = re.compile(re.escape("__SQL_PLACEHOLDER__"))
        sql_parts = placeholder_pattern.split(formatted_sql)
        new_values = []
        num_placeholders = len(format_values)

        # Ensure the number of placeholders matches
        if len(sql_parts) - 1 != num_placeholders:
            logger.debug(
                "Mismatch between placeholders and format values. Skipping formatting.",
            )
            return None  # Skip formatting

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


def format_sql_magic(
    code,
    sql_keywords,
    sqlfluff_config_path,
    dialect,
    sqlfluff_rules="",
    sqlfluff_fix_args="",
):
    """Formats SQL magic commands within a code cell.
    Handles both %%sql (cell magic) and %sql (line magic).
    """
    lines = code.split("\n")
    new_lines = []
    in_sql_magic = False
    sql_code = ""
    magic_line = ""
    changed = False  # Track if any changes were made
    for line in lines:
        stripped_line = line.strip()
        if re.match(r"^%%sql\b", stripped_line, re.IGNORECASE):
            in_sql_magic = True
            magic_line = line
            sql_code = ""
            continue
        if re.match(r"^%sql\b", stripped_line, re.IGNORECASE):
            # Line magic; assume rest of the line is SQL code
            # Handle potential parameters/options after %sql
            match = re.match(r"^%sql\b\s*(.*)", stripped_line, re.IGNORECASE)
            if match:
                sql_line = match.group(1).strip()
                if sql_line:
                    if contains_sql_keywords(sql_line, sql_keywords):
                        formatted_sql = format_sql_code(
                            sql_line,
                            sqlfluff_config_path,
                            dialect,
                            sqlfluff_rules,
                            sqlfluff_fix_args,
                        )
                        if formatted_sql and formatted_sql.strip() != sql_line.strip():
                            # Preserve any parameters/options before the SQL
                            prefix = line[: line.lower().find("%sql") + 4]
                            new_lines.append(prefix + " " + formatted_sql.strip())
                            changed = True
                        else:
                            new_lines.append(line)
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
                if contains_sql_keywords(sql_code, sql_keywords):
                    formatted_sql = format_sql_code(
                        sql_code,
                        sqlfluff_config_path,
                        dialect,
                        sqlfluff_rules,
                        sqlfluff_fix_args,
                    )
                    if formatted_sql and formatted_sql.strip() != sql_code.strip():
                        new_lines.append(magic_line)
                        new_lines.extend(formatted_sql.strip().split("\n"))
                        changed = True
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
        if contains_sql_keywords(sql_code, sql_keywords):
            formatted_sql = format_sql_code(
                sql_code,
                sqlfluff_config_path,
                dialect,
                sqlfluff_rules,
                sqlfluff_fix_args,
            )
            if formatted_sql and formatted_sql.strip() != sql_code.strip():
                new_lines.append(magic_line)
                new_lines.extend(formatted_sql.strip().split("\n"))
                changed = True
            else:
                new_lines.append(magic_line)
                new_lines.extend(sql_code.strip().split("\n"))
        else:
            new_lines.append(magic_line)
            new_lines.extend(sql_code.strip().split("\n"))
    new_code = "\n".join(new_lines)
    return new_code, changed


def contains_sql_keywords(sql_code, sql_keywords):
    """Determines if the given code contains at least two SQL keywords."""
    upper_sql = sql_code.upper()
    # Exclude strings that start with common file path indicators or URLs
    if re.match(r"^\s*(/|[a-zA-Z]:\\|https?://)", sql_code.strip()):
        return False
    # Use word boundaries to match whole words
    found_keywords = [
        keyword
        for keyword in sql_keywords
        if re.search(r"\b" + re.escape(keyword) + r"\b", upper_sql)
    ]
    return len(found_keywords) >= 2  # Require at least two keywords


def format_sql_in_code(
    code,
    sqlfluff_config_path,
    dialect,
    sqlfluff_rules="",
    sqlfluff_fix_args="",
):
    """Parses and formats SQL strings within Python code."""
    try:
        tree = ast.parse(code)
        formatter = SQLStringFormatter(
            sql_keywords,
            function_names,
            sqlfluff_config_path,
            dialect,
            sqlfluff_rules,
            sqlfluff_fix_args,
        )
        formatter.visit(tree)
        if not formatter.changed:
            # No changes were made to the AST
            return code
        try:
            formatted_code = ast.unparse(tree)
        except AttributeError:
            formatted_code = astor.to_source(tree)
        return formatted_code
    except Exception as e:
        logger.debug(f"Error parsing code: {e}")
        return code


def format_sql_in_notebook(
    notebook_path,
    sql_keywords,
    function_names,
    sqlfluff_config_path,
    dialect,
    sqlfluff_rules="",
    sqlfluff_fix_args="",
):
    """Processes a Jupyter notebook to format SQL code within it."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.cell_type == "code":
                original_code = cell.source
                logger.debug(f"Processing cell:\n{original_code}")

                # Detect SQL magic commands with improved regex
                has_sql_magic = bool(
                    re.search(
                        r"^\s*%%sql\b|^\s*%sql\b",
                        original_code,
                        re.IGNORECASE | re.MULTILINE,
                    ),
                )

                # Detect SQL function calls
                has_sql_functions = bool(
                    re.search(
                        r"\b(" + "|".join(map(re.escape, function_names)) + r")\b",
                        original_code,
                        re.IGNORECASE,
                    ),
                )

                # Detect SQL string assignments (including f-strings)
                has_sql_strings = False
                # Parse the AST to check for SQL string assignments
                try:
                    tree = ast.parse(original_code)
                    formatter = SQLStringFormatter(
                        sql_keywords,
                        function_names,
                        sqlfluff_config_path,
                        dialect,
                        sqlfluff_rules,
                        sqlfluff_fix_args,
                    )
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Assign):
                            if isinstance(node.value, (ast.Constant, ast.JoinedStr)):
                                sql_code, _ = formatter.extract_sql_from_string(
                                    node.value,
                                )
                                if formatter.contains_sql_keywords(sql_code):
                                    has_sql_strings = True
                                    break
                except Exception as e:
                    logger.debug(f"Error parsing cell AST: {e}")

                # Determine if the cell contains SQL code
                if has_sql_magic or has_sql_functions or has_sql_strings:
                    logger.info("SQL code detected in cell. Formatting...")

                    # First, format SQL magic commands
                    code_with_formatted_magic, magic_changed = format_sql_magic(
                        original_code,
                        sql_keywords,
                        sqlfluff_config_path,
                        dialect,
                        sqlfluff_rules,
                        sqlfluff_fix_args,
                    )

                    # Then, format SQL within Python code
                    formatted_code = format_sql_in_code(
                        code_with_formatted_magic,
                        sqlfluff_config_path,
                        dialect,
                        sqlfluff_rules,
                        sqlfluff_fix_args,
                    )

                    # Add logging statements to compare original and formatted code
                    logger.debug(f"Original code:\n{original_code}")
                    logger.debug(f"Formatted code:\n{formatted_code}")

                    code_changed = formatted_code != code_with_formatted_magic
                    if code_changed or magic_changed:
                        # Show diff for transparency
                        logger.info("Changes detected in SQL cell. Showing diff:")
                        diff = difflib.unified_diff(
                            original_code.splitlines(),
                            formatted_code.splitlines(),
                            fromfile="before.py",
                            tofile="after.py",
                            lineterm="",
                        )
                        diff_text = "\n".join(diff)
                        if diff_text:
                            logger.info(diff_text)

                        # Update the cell source
                        cell.source = formatted_code
                        changed = True
                    else:
                        logger.debug("No changes detected in SQL cell.")
                else:
                    logger.debug("No SQL code detected in cell. Skipping formatting.")
        if changed:
            nbformat.write(nb, notebook_path)
            logger.info(f"Formatted SQL code in {notebook_path}")
            return True
        logger.info(f"No SQL code changes in {notebook_path}")
        return False
    except Exception as e:
        logger.error(f"Error processing notebook {notebook_path}: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format SQL code in Jupyter notebooks.",
    )
    parser.add_argument(
        "notebooks",
        metavar="notebook",
        type=str,
        nargs="+",
        help="Path(s) to Jupyter notebook(s) to process.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the script configuration YAML file.",
    )
    parser.add_argument(
        "--sqlfluff-config",
        type=str,
        default=".sqlfluff",
        help="Path to the sqlfluff configuration INI file.",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="ansi",
        help="SQL dialect to use with sqlfluff (e.g., ansi, mysql, postgres, etc.).",
    )
    parser.add_argument(
        "--sqlfluff-rules",
        type=str,
        default="",
        help="Comma-separated list of sqlfluff rules to apply.",
    )
    parser.add_argument(
        "--sqlfluff-fix-args",
        type=str,
        default="",
        help="Additional arguments to pass to sqlfluff fix.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    args = parser.parse_args()

    # Set up logging based on the log level
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))

    # Load script-specific configuration
    sql_keywords, function_names = load_config(args.config)

    # Determine the absolute path to .sqlfluff
    sqlfluff_config_path = None
    if args.sqlfluff_config:
        sqlfluff_config_path = Path(args.sqlfluff_config)
        if not sqlfluff_config_path.is_file():
            logger.error(
                f"sqlfluff configuration file '{sqlfluff_config_path}' not found.",
            )
            sys.exit(1)

    any_notebook_changed = False
    for notebook_path in args.notebooks:
        logger.info(f"Processing notebook: {notebook_path}")
        if format_sql_in_notebook(
            notebook_path,
            sql_keywords,
            function_names,
            sqlfluff_config_path,
            args.dialect,
            args.sqlfluff_rules,
            args.sqlfluff_fix_args,
        ):
            any_notebook_changed = True

    if any_notebook_changed:
        logger.info("SQL formatting changes were made.")
        sys.exit(1)  # Indicate that changes were made
    else:
        logger.info("No SQL formatting changes were necessary.")
        sys.exit(0)  # Indicate that no changes were made
