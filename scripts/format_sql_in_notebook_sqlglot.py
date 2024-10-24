import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set, Optional, Dict, Union

import astor
import nbformat
import yaml
from sqlglot import parse_one, errors


@dataclass
class FormattingConfig:
    """Configuration for SQL formatting."""

    sql_keywords: Set[str]
    function_names: Set[str]
    sql_decorators: Set[str]
    single_line_threshold: int = 80
    preserve_comments: bool = True
    indent_width: int = 4


class SQLFormattingError(Exception):
    """Custom exception for SQL formatting errors."""

    pass


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Sets up logging with the specified level."""
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_config(config_path: Union[str, Path] = "config.yaml") -> FormattingConfig:
    """Loads configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    try:
        with open(config_file) as file:
            config = yaml.safe_load(file)
            return FormattingConfig(
                sql_keywords=set(config.get("sql_keywords", [])),
                function_names=set(config.get("function_names", [])),
                sql_decorators=set(config.get("sql_decorators", [])),
                single_line_threshold=config.get("formatting_options", {}).get(
                    "single_line_threshold", 80
                ),
                preserve_comments=config.get("formatting_options", {}).get(
                    "preserve_comments", True
                ),
                indent_width=config.get("formatting_options", {}).get(
                    "indent_width", 4
                ),
            )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing the configuration file: {e}")


def format_sql_code(
    sql_code: str,
    dialect: Optional[str],
    config: FormattingConfig,
    placeholders: Optional[Dict[str, str]] = None,
    force_single_line: bool = False,
    is_magic_command: bool = False,
    is_cell_magic: bool = False,
) -> str:
    """Formats SQL code using sqlglot's native formatting capabilities."""
    try:
        # Initialize logger within the function
        logger = logging.getLogger(__name__)

        if not sql_code.strip():
            return sql_code

        # Preserve comments if configured
        comments = []
        if config.preserve_comments:
            comment_pattern = r"(--[^\n]*|/\*.*?\*/)"
            comments = list(re.finditer(comment_pattern, sql_code, re.DOTALL))
            temp_sql = re.sub(comment_pattern, "__COMMENT__", sql_code)
        else:
            temp_sql = sql_code

        # Handle placeholders
        if placeholders:
            for pos, expr in placeholders.items():
                temp_sql = temp_sql.replace(pos, f"__TEMP_PLACEHOLDER_{expr}__")

        temp_sql = temp_sql.strip()

        # Handle 'ORDER BY n' and 'GROUP BY n' cases
        select_match = re.search(
            r"SELECT\s+(.*?)\s+FROM", temp_sql, re.IGNORECASE | re.DOTALL
        )
        if select_match:
            select_items = [item.strip() for item in select_match.group(1).split(",")]

            for clause in ["GROUP BY", "ORDER BY"]:
                pattern = rf"{clause}\s+(\d+)"
                match = re.search(pattern, temp_sql, re.IGNORECASE)
                if match:
                    col_num = int(match.group(1)) - 1
                    if 0 <= col_num < len(select_items):
                        temp_sql = re.sub(
                            pattern,
                            f"{clause} {select_items[col_num]}",
                            temp_sql,
                            flags=re.IGNORECASE,
                        )

        # Parse and format SQL
        parsed = parse_one(temp_sql, read=dialect)
        formatted_sql = parsed.sql(pretty=True)

        # Apply formatting based on context
        if is_magic_command and not is_cell_magic:
            # Line magic: single line
            formatted_sql = " ".join(formatted_sql.split())
        elif force_single_line:
            formatted_sql = " ".join(formatted_sql.split())
        else:
            # For both magic commands and regular strings, preserve sqlglot's formatting
            formatted_sql = formatted_sql.strip()

        # Restore placeholders
        if placeholders:
            for expr in placeholders.values():
                formatted_sql = formatted_sql.replace(
                    f"__TEMP_PLACEHOLDER_{expr}__", "{" + expr + "}"
                )

        # Restore comments
        if config.preserve_comments and comments:
            for comment in comments:
                formatted_sql = formatted_sql.replace(
                    "__COMMENT__", comment.group(0), 1
                )

        # Debug Logging
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return formatted_sql

    except errors.ParseError as e:
        raise SQLFormattingError(f"Failed to parse SQL code: {e}")
    except Exception as e:
        raise SQLFormattingError(f"Unexpected error during SQL formatting: {e}")


class SQLStringFormatter(ast.NodeTransformer):
    """AST NodeTransformer that formats SQL strings."""

    def __init__(
        self, config: FormattingConfig, dialect: Optional[str], logger: logging.Logger
    ):
        super().__init__()
        self.config = config
        self.dialect = dialect
        self.logger = logger
        self.changed = False

    def is_likely_sql(self, code: str) -> bool:
        """Enhanced SQL detection with better heuristics."""
        if not code or len(code.strip()) < 10:
            return False

        if re.match(r"^\s*(/|[a-zA-Z]:\\|https?://|<!DOCTYPE|<html)", code.strip()):
            return False

        upper_code = code.upper()
        keyword_count = sum(
            1
            for keyword in self.config.sql_keywords
            if re.search(rf"\b{re.escape(keyword)}\b", upper_code)
        )

        has_sql_pattern = bool(
            re.search(
                r"\bSELECT\b.*\bFROM\b|\bUPDATE\b.*\bSET\b|\bINSERT\b.*\bINTO\b|\bDELETE\b.*\bFROM\b",
                upper_code,
                re.DOTALL,
            )
        )

        return keyword_count >= 2 or has_sql_pattern

    def extract_fstring_parts(self, node: ast.JoinedStr) -> tuple[str, Dict[str, str]]:
        """Extracts parts of an f-string, preserving expressions."""
        sql_parts = []
        placeholders = {}
        current_pos = 0

        for value in node.values:
            if isinstance(value, ast.Constant):
                sql_parts.append(value.value)
                current_pos += len(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = astor.to_source(value.value).strip()
                placeholder = f"__PLACEHOLDER_{current_pos}__"
                sql_parts.append(placeholder)
                placeholders[placeholder] = expr
                current_pos += len(placeholder)

        return "".join(sql_parts), placeholders

    def format_sql_node(
        self, node: Union[ast.Constant, ast.JoinedStr], force_single_line: bool = False
    ) -> Optional[ast.AST]:
        """Formats SQL code in AST nodes."""
        try:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if not self.is_likely_sql(node.value):
                    return None
                formatted_sql = format_sql_code(
                    node.value,
                    self.dialect,
                    self.config,
                    force_single_line=force_single_line,
                )
                if formatted_sql != node.value:
                    self.changed = True
                    # Check if the string is multi-line
                    if "\n" in formatted_sql:
                        # Reconstruct the multi-line string with newlines after opening and before closing quotes
                        reconstructed_str = f'"""\n{formatted_sql}\n"""'
                        formatted_node = ast.parse(reconstructed_str).body[0].value
                    else:
                        # Single-line string
                        reconstructed_str = f'"""{formatted_sql}"""'
                        formatted_node = ast.parse(reconstructed_str).body[0].value
                    return formatted_node

            elif isinstance(node, ast.JoinedStr):
                sql_str, placeholders = self.extract_fstring_parts(node)
                if not self.is_likely_sql(sql_str):
                    return None
                formatted_sql = format_sql_code(
                    sql_str,
                    self.dialect,
                    self.config,
                    placeholders=placeholders,
                    force_single_line=force_single_line,
                )
                if formatted_sql != sql_str:
                    self.changed = True
                    # Reconstruct the f-string with newlines after opening and before closing quotes
                    if "\n" in formatted_sql:
                        reconstructed_fstring = f"\n{formatted_sql}\n"
                    else:
                        reconstructed_fstring = formatted_sql
                    formatted_node = (
                        ast.parse(f'f"""{reconstructed_fstring}"""').body[0].value
                    )
                    return formatted_node
            return None
        except SQLFormattingError as e:
            self.logger.warning(f"SQL formatting skipped: {e}")
            return None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handles assignments."""
        if isinstance(node.value, (ast.Constant, ast.JoinedStr)):
            formatted_node = self.format_sql_node(node.value)
            if formatted_node:
                node.value = formatted_node
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handles function calls."""
        func_name = self.get_full_func_name(node.func)
        if any(name in func_name for name in self.config.function_names):
            for idx, arg in enumerate(node.args):
                if isinstance(arg, (ast.Constant, ast.JoinedStr)):
                    formatted_node = self.format_sql_node(arg)
                    if formatted_node:
                        node.args[idx] = formatted_node
            for keyword in node.keywords:
                if isinstance(keyword.value, (ast.Constant, ast.JoinedStr)):
                    formatted_node = self.format_sql_node(keyword.value)
                    if formatted_node:
                        keyword.value = formatted_node
        return self.generic_visit(node)

    @staticmethod
    def get_full_func_name(node: ast.AST) -> str:
        """Gets the full function name from an AST node."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


def process_notebook(
    notebook_path: Union[str, Path],
    config: FormattingConfig,
    dialect: Optional[str],
    logger: logging.Logger,
) -> bool:
    """Processes a Jupyter notebook."""
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
        changed = False

        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue

            original_code = cell.source
            if not original_code.strip():
                continue

            lines = original_code.split("\n")

            # Initialize variables to track magic commands
            magic_cmd = None
            magic_cmd_index = None

            # Iterate through lines to find the first non-comment magic command
            for idx, line in enumerate(lines):
                stripped = line.strip()

                if not stripped:
                    continue  # Skip empty lines

                if stripped.startswith("#"):
                    continue  # Skip comment lines

                if stripped.startswith("%%sql") or stripped.startswith("%sql"):
                    magic_cmd = stripped.split()[0]
                    magic_cmd_index = idx
                    break  # Magic command found

                else:
                    break  # Non-magic, non-comment line found

            if magic_cmd:
                is_cell_magic = magic_cmd.startswith("%%sql")

                if is_cell_magic:
                    # Cell magic: SQL code starts from the next line
                    sql_code = "\n".join(lines[magic_cmd_index + 1 :]).strip()
                else:
                    # Line magic: SQL code is on the same line after the magic command
                    sql_code = lines[magic_cmd_index][len(magic_cmd) :].strip()

                try:
                    formatted_sql = format_sql_code(
                        sql_code,
                        dialect,
                        config,
                        is_magic_command=True,
                        is_cell_magic=is_cell_magic,
                    )

                    # Reconstruct the cell content
                    if is_cell_magic:
                        # Preserve comments before the magic command
                        pre_magic = "\n".join(lines[:magic_cmd_index])
                        if pre_magic:
                            new_content = f"{pre_magic}\n{magic_cmd}\n{formatted_sql}"
                        else:
                            new_content = f"{magic_cmd}\n{formatted_sql}"
                    else:
                        # Preserve comments before the magic command
                        pre_magic = "\n".join(lines[:magic_cmd_index])
                        if pre_magic:
                            new_content = f"{pre_magic}\n{magic_cmd} {formatted_sql}"
                        else:
                            new_content = f"{magic_cmd} {formatted_sql}"

                    if new_content != original_code:
                        cell.source = new_content
                        changed = True

                except SQLFormattingError as e:
                    logger.warning(f"SQL magic formatting skipped: {e}")

                continue  # Move to the next cell after handling magic command

            # Handle regular Python code
            try:
                tree = ast.parse(original_code)
                formatter = SQLStringFormatter(config, dialect, logger)
                new_tree = formatter.visit(tree)
                if formatter.changed:
                    # Use astor to convert AST back to source code
                    formatted_code = astor.to_source(new_tree)
                    if formatted_code != original_code:
                        cell.source = formatted_code
                        changed = True
            except SyntaxError:
                logger.warning(f"Failed to parse cell:\n{original_code}")
                continue

        if changed:
            nbformat.write(notebook, notebook_path)
            logger.info(f"Updated notebook: {notebook_path}")

        return changed

    except Exception as e:
        logger.error(f"Failed to process notebook {notebook_path}: {e}")
        raise


def main():
    """Main entry point for the SQL formatter."""
    import argparse

    parser = argparse.ArgumentParser(description="Format SQL code in Jupyter notebooks")
    parser.add_argument(
        "notebooks", nargs="+", type=Path, help="Notebook paths to process"
    )
    parser.add_argument(
        "--config", type=Path, default="config.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--dialect", type=str, help="SQL dialect (e.g., mysql, postgres)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        changed = False

        for notebook in args.notebooks:
            if process_notebook(notebook, config, args.dialect, logger):
                changed = True

        sys.exit(1 if changed else 0)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
