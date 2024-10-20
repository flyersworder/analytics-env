import nbformat
import re
import sqlparse
import logging
import sys
import difflib
import argparse
import yaml
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Default logging level
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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
            return sql_keywords
    except yaml.YAMLError as e:
        logger.error(f"Error parsing the configuration file: {e}")
        sys.exit(1)


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


def format_sql_code(sql_code):
    """Formats the given SQL code using sqlparse with the specified compact configuration."""
    formatted_sql = sqlparse.format(
        sql_code,
        reindent=True,
        keyword_case="upper",
        identifier_case="lower",
        use_space_around_operators=True,
        comma_first=False,
        indent_columns=True,
        wrap_after=1,  # Each column on a new line
        indent_width=4,
        compact=False,
    )
    return formatted_sql


def format_magic_commands(code, sql_keywords):
    """
    Formats SQL within magic commands (%%sql and %sql).
    - %%sql: Formats multi-line SQL statements.
    - %sql: Formats single-line SQL statements.
    """
    lines = code.split("\n")
    new_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        stripped = line.strip()

        # Handle %%sql magic
        if stripped.startswith("%%sql"):
            magic_command = stripped
            sql_lines = []
            # Collect all lines after %%sql until the end of the cell
            for sql_line in lines[i + 1 :]:
                if sql_line.strip().startswith("%%") or sql_line.strip().startswith(
                    "%"
                ):
                    # Another magic command starts; stop collecting
                    break
                sql_lines.append(sql_line)
            sql_code = "\n".join(sql_lines).strip()
            if contains_sql_keywords(sql_code, sql_keywords):
                formatted_sql = format_sql_code(sql_code)
                formatted_cell = f"{magic_command}\n{formatted_sql}"
                new_lines.append(formatted_cell)
                # Skip the SQL lines as they have been processed
                skip_next = True
            else:
                new_lines.append(line)
        # Handle %sql magic
        elif stripped.startswith("%sql"):
            parts = line.split("%sql", 1)
            if len(parts) == 2:
                magic, sql_code = parts
                sql_code = sql_code.strip()
                if contains_sql_keywords(sql_code, sql_keywords):
                    formatted_sql = format_sql_code(sql_code)
                    # For line magic, keep SQL on a single line by joining
                    formatted_sql_single_line = " ".join(
                        formatted_sql.strip().splitlines()
                    )
                    new_line = f"{magic}%sql {formatted_sql_single_line}"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def format_assignments(code, sql_keywords):
    """
    Formats SQL within variable assignments and function calls.
    - Variable Assignments: query = "SELECT ...", including multi-line.
    - Function Calls: execute_query("SELECT ...")
    """

    # Handle variable assignments
    def assignment_replacer(match):
        indent = match.group("indent")
        var_name = match.group("var_name")
        quote_prefix = match.group("quote_prefix")
        quote_char = match.group("quote_char")
        sql_code = match.group("sql_code")

        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return original

        # Determine if it's an f-string
        is_f_string = "f" in quote_prefix.lower()

        # Format the SQL code
        formatted_sql = format_sql_code(sql_code)

        # Wrap the formatted SQL code in triple quotes
        if quote_char.startswith('"'):
            triple_quote_char = '"""'
        else:
            triple_quote_char = "'''"

        # Check if the original assignment was multi-line (enclosed in parentheses)
        multi_line_assignment = bool(re.search(r"\(\s*$", match.group(0), re.MULTILINE))

        if multi_line_assignment:
            # For multi-line assignments, place triple quotes on new lines
            formatted_sql_wrapped = (
                f"{triple_quote_char}\n{formatted_sql}\n{triple_quote_char}"
            )
        else:
            # For single-line assignments, place triple quotes inline
            formatted_sql_wrapped = (
                f"{triple_quote_char}{formatted_sql}{triple_quote_char}"
            )

        # Reconstruct the assignment
        new_quote_prefix = quote_prefix.replace("f", "").replace(
            "F", ""
        )  # Remove 'f' from prefix
        if is_f_string:
            new_quote_prefix = "f" + new_quote_prefix
        new_line = f"{indent}{var_name} = {new_quote_prefix}{formatted_sql_wrapped}"

        logger.debug(f"Formatting assignment for variable '{var_name}'.")
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return new_line

    assignment_pattern = re.compile(
        r"""
        (?P<indent>^[ \t]*)                # Indentation at the start of the line
        (?P<var_name>\w+)                  # Variable name
        [ \t]*=[ \t]*                      # Assignment operator
        (?P<quote_prefix>[frbuFRBU]*)      # Optional prefixes (f, r, b, u)
        (?P<quote_char>['"]{1,3})          # Opening quote(s)
        (?P<sql_code>(?:\\.|(?!\4).)*?)    # SQL code, handling escaped quotes
        (?P=quote_char)                    # Closing quote(s) matching opening
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    code = assignment_pattern.sub(assignment_replacer, code)

    # Handle function calls with SQL string arguments
    def function_call_replacer(match):
        indent = match.group("indent")
        func_name = match.group("func_name")
        quote_prefix = match.group("quote_prefix")
        quote_char = match.group("quote_char")
        sql_code = match.group("sql_code")

        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return original

        # Determine if it's an f-string
        is_f_string = "f" in quote_prefix.lower()

        # Format the SQL code
        formatted_sql = format_sql_code(sql_code)

        # Wrap the formatted SQL code in triple quotes
        if quote_char.startswith('"'):
            triple_quote_char = '"""'
        else:
            triple_quote_char = "'''"

        # Check if the original function call was multi-line (enclosed in parentheses)
        multi_line_call = bool(re.search(r"\(\s*$", match.group(0), re.MULTILINE))

        if multi_line_call:
            # For multi-line function calls, place triple quotes on new lines
            formatted_sql_wrapped = (
                f"{triple_quote_char}\n{formatted_sql}\n{triple_quote_char}"
            )
        else:
            # For single-line function calls, place triple quotes inline
            formatted_sql_wrapped = (
                f"{triple_quote_char}{formatted_sql}{triple_quote_char}"
            )

        # Reconstruct the function call
        new_quote_prefix = quote_prefix.replace("f", "").replace(
            "F", ""
        )  # Remove 'f' from prefix
        if is_f_string:
            new_quote_prefix = "f" + new_quote_prefix
        new_line = f"{indent}{func_name}({new_quote_prefix}{formatted_sql_wrapped})"

        logger.debug(f"Formatting function call '{func_name}'.")
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        return new_line

    function_call_pattern = re.compile(
        r"""
        (?P<indent>^[ \t]*)                # Indentation
        (?P<func_name>\w+)                  # Function name
        [ \t]*\([ \t]*                      # Opening parenthesis
        (?P<quote_prefix>[frbuFRBU]*)       # Optional string prefixes
        (?P<quote_char>['"]{1,3})           # Opening quote(s)
        (?P<sql_code>(?:\\.|(?!\4).)*?)     # SQL code
        (?P=quote_char)                     # Closing quote(s) matching opening
        [ \t]*\)                            # Closing parenthesis
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    code = function_call_pattern.sub(function_call_replacer, code)

    return code


def format_notebook_cell(code, sql_keywords):
    """Formats a single notebook code cell."""
    original_code = code
    logger.debug(f"Original code:\n{original_code}")

    # Step 1: Format magic commands
    code = format_magic_commands(code, sql_keywords)
    logger.debug(f"After formatting magic commands:\n{code}")

    # Step 2: Format variable assignments and function calls
    code = format_assignments(code, sql_keywords)
    logger.debug(f"After formatting assignments and function calls:\n{code}")

    # Determine if any changes were made
    changes_made = code != original_code

    return code, changes_made


def format_notebook(notebook_path, sql_keywords):
    """Processes a Jupyter notebook to format SQL code within it."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        notebook_changed = False

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                original_code = cell.source
                logger.debug(f"Processing cell {idx + 1}:\n{original_code}")

                # Format the code cell
                formatted_code, changed = format_notebook_cell(
                    original_code, sql_keywords
                )

                if changed:
                    # Show diff for transparency
                    logger.info(f"Changes detected in cell {idx + 1}. Showing diff:")
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
                    notebook_changed = True
                else:
                    logger.debug(f"No changes detected in cell {idx + 1}.")

        if notebook_changed:
            nbformat.write(nb, notebook_path)
            logger.info(f"Formatted SQL code in {notebook_path}")
            return True
        else:
            logger.info(f"No SQL formatting changes were necessary in {notebook_path}.")
            return False
    except Exception as e:
        logger.error(f"Error processing notebook {notebook_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Format SQL code in Jupyter notebooks by replacing SQL strings and magic commands with formatted versions."
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
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Update logging level based on user input
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))

    # Load SQL keywords from configuration
    sql_keywords = load_config(args.config)

    any_notebook_changed = False
    for notebook_path in args.notebooks:
        logger.info(f"Processing notebook: {notebook_path}")
        if format_notebook(notebook_path, sql_keywords):
            any_notebook_changed = True

    if any_notebook_changed:
        logger.info("SQL formatting changes were made.")
        sys.exit(1)  # Indicate that changes were made
    else:
        logger.info("No SQL formatting changes were necessary.")
        sys.exit(0)  # Indicate that no changes were made


if __name__ == "__main__":
    main()
