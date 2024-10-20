import ast
import difflib
import logging
import re
import sys
from pathlib import Path

import astor
import nbformat
import sqlparse
import yaml

# Determine if we're using Python 3.8 or higher
PY38_PLUS = sys.version_info >= (3, 8)

# Set up logging
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Loads the configuration from a YAML file."""
    config_file = Path(config_path)
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


def format_sql_code(sql_code):
    """Formats the given SQL code using sqlparse."""
    formatted_sql = sqlparse.format(
        sql_code,
        reindent=True,
        keyword_case="upper",
        identifier_case="lower",
        use_space_around_operators=True,
        comma_first=False,
        indent_columns=True,
        wrap_after=1,  # This will ensure each column is on a new line
        indent_width=4,
        compact=True,
    )
    return formatted_sql


class SQLStringFormatter(ast.NodeTransformer):
    """AST NodeTransformer that formats SQL strings in Assign and Call nodes."""

    def __init__(self, sql_keywords, function_names):
        super().__init__()
        self.sql_keywords = sql_keywords
        self.function_names = function_names
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
        if self.is_string_node(node.value):
            sql_code = self.get_string_value(node.value)
            if not sql_code:
                return self.generic_visit(node)
            logger.debug(f"Examining potential SQL code in assignment:\n{sql_code}")
            if self.contains_sql_keywords(sql_code):
                logger.info(f"SQL keywords detected in assignment:\n{sql_code}")
                try:
                    formatted_sql = format_sql_code(sql_code)
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
                        # Set the formatted SQL code in the node
                        if PY38_PLUS:
                            node.value = ast.Constant(value=formatted_sql)
                        else:
                            node.value = ast.Str(s=formatted_sql)
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
                if self.is_string_node(arg):
                    sql_code = self.get_string_value(arg)
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
                            formatted_sql = format_sql_code(sql_code)
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
                                # Set the formatted SQL code in the argument
                                if PY38_PLUS:
                                    node.args[idx] = ast.Constant(value=formatted_sql)
                                else:
                                    node.args[idx] = ast.Str(s=formatted_sql)
                                self.changed = True  # Indicate that a change was made
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


def format_sql_magic(code, sql_keywords):
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
                        formatted_sql = format_sql_code(sql_line)
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
                    formatted_sql = format_sql_code(sql_code)
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
            formatted_sql = format_sql_code(sql_code)
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


def format_sql_in_code(code):
    """Parses and formats SQL strings within Python code."""
    try:
        tree = ast.parse(code)
        formatter = SQLStringFormatter(sql_keywords, function_names)
        formatter.visit(tree)
        if not formatter.changed:
            # No changes were made to the AST
            return code

        # Custom code generator to control string representation
        class CustomCodeGenerator(astor.CodeGenerator):
            def visit_Str(self, node):
                # Use triple quotes for strings with newlines
                if "\n" in node.s:
                    self.write('"""')
                    self.write(node.s)
                    self.write('"""')
                else:
                    super().visit_Str(node)

            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    # Use triple quotes for strings with newlines
                    if "\n" in node.value:
                        self.write('"""')
                        self.write(node.value)
                        self.write('"""')
                    else:
                        super().visit_Constant(node)
                else:
                    super().visit_Constant(node)

        # Generate the formatted code using the custom code generator
        formatted_code = astor.to_source(tree, code_gen=CustomCodeGenerator)

        return formatted_code
    except Exception as e:
        logger.debug(f"Error parsing code: {e}")
        return code


def format_sql_in_notebook(notebook_path, sql_keywords, function_names):
    """Processes a Jupyter notebook to format SQL code within it."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.cell_type == "code":
                original_code = cell.source
                logger.debug(f"Processing cell:\n{original_code}")

                # Format SQL magic commands
                code_with_formatted_magic, magic_changed = format_sql_magic(
                    original_code,
                    sql_keywords,
                )

                # Format SQL within Python code
                formatted_code = format_sql_in_code(code_with_formatted_magic)

                # Determine if any changes were made
                code_changed = formatted_code != original_code

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
        if changed:
            nbformat.write(nb, notebook_path)
            logger.info(f"Formatted SQL code in {notebook_path}")
            return True
        logger.info("No SQL formatting changes were necessary.")
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

    any_notebook_changed = False
    for notebook_path in args.notebooks:
        logger.info(f"Processing notebook: {notebook_path}")
        if format_sql_in_notebook(
            notebook_path,
            sql_keywords,
            function_names,
        ):
            any_notebook_changed = True

    if any_notebook_changed:
        logger.info("SQL formatting changes were made.")
        sys.exit(1)  # Indicate that changes were made
    else:
        logger.info("No SQL formatting changes were necessary.")
        sys.exit(0)  # Indicate that no changes were made
