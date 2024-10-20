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
logger.setLevel(logging.DEBUG)  # Set default logging level to DEBUG
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
    """Formats the given SQL code using sqlparse with the specified configuration."""
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


def format_sql_in_code_cell(code, sql_keywords):
    """Formats SQL strings assigned to variables and SQL magic commands in a code cell."""
    changed = False

    # Regular expression to match variable assignments with strings (including f-strings)
    assignment_pattern = re.compile(
        r"""
        (?P<indent>^[ \t]*)                # Indentation at the start of the line
        (?P<var_name>\w+)                  # Variable name
        [ \t]*=[ \t]*                      # Assignment operator
        (?P<quote_prefix>[frbuFRBU]*)      # Optional prefixes (f, r, b, u)
        (?P<quote_char>['"]{1,3})          # Opening quote(s)
        (?P<sql_code>.*?)                   # SQL code (non-greedy)
        (?P=quote_char)                    # Closing quote(s) matching opening
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    def assignment_replacer(match):
        nonlocal changed
        indent = match.group("indent")
        var_name = match.group("var_name")
        quote_prefix = match.group("quote_prefix")
        quote_char = match.group("quote_char")
        sql_code = match.group("sql_code")

        # Check if the string contains SQL keywords
        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return the original string

        # Handle f-strings: preserve placeholders
        is_f_string = "f" in quote_prefix.lower()
        if is_f_string:
            # Find placeholders in the f-string
            placeholder_pattern = re.compile(r"(\{[^}]+\})")
            placeholders = placeholder_pattern.findall(sql_code)
            placeholder_markers = [
                f"__PLACEHOLDER_{i}__" for i in range(len(placeholders))
            ]

            # Replace placeholders with markers
            sql_code_no_placeholders = sql_code
            for ph, marker in zip(placeholders, placeholder_markers):
                sql_code_no_placeholders = sql_code_no_placeholders.replace(ph, marker)

            # Format the SQL code without placeholders
            formatted_sql_no_placeholders = format_sql_code(sql_code_no_placeholders)

            # Replace markers with placeholders
            formatted_sql = formatted_sql_no_placeholders
            for ph, marker in zip(placeholders, placeholder_markers):
                formatted_sql = formatted_sql.replace(marker, ph)
        else:
            # Not an f-string, format directly
            formatted_sql = format_sql_code(sql_code)

        # Wrap the formatted SQL code in triple quotes
        triple_quote_char = '"""' if '"' in quote_char else "'''"
        formatted_sql_wrapped = f"{triple_quote_char}{formatted_sql}{triple_quote_char}"

        # Reconstruct the assignment
        new_quote_prefix = quote_prefix.replace("f", "").replace(
            "F", ""
        )  # Remove 'f' from prefix
        if is_f_string:
            new_quote_prefix = "f" + new_quote_prefix
        new_line = f"{indent}{var_name} = {new_quote_prefix}{formatted_sql_wrapped}"

        # Log the change
        logger.debug(f"Formatting assignment for variable '{var_name}'.")
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        changed = True
        return new_line

    # Replace all variable assignments with formatted SQL
    formatted_code, num_subs = assignment_pattern.subn(assignment_replacer, code)

    # Now handle SQL magic commands
    # Handle %%sql (cell magic)
    cell_magic_pattern = re.compile(
        r"""
        ^(?P<magic>%%sql\b.*\n)           # %%sql magic command
        (?P<sql>(?:.*\n)*?)               # SQL code (non-greedy)
        (?=^\S|\Z)                         # Lookahead for non-indented line or end of string
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def cell_magic_replacer(match):
        nonlocal changed
        magic = match.group("magic")
        sql_code = match.group("sql")

        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return original

        # Format the SQL code
        formatted_sql = format_sql_code(sql_code)

        # Remove leading indentation from each line
        formatted_sql = "\n".join(line.lstrip() for line in formatted_sql.split("\n"))

        # Reconstruct the magic command
        new_magic = f"{magic.strip()}\n{formatted_sql}\n"

        logger.debug("Formatting %%sql magic command.")
        logger.debug(f"Original SQL:\n{sql_code}")
        logger.debug(f"Formatted SQL:\n{formatted_sql}")

        changed = True
        return new_magic

    # Replace all %%sql magic commands with formatted SQL
    formatted_code, num_subs_magic = cell_magic_pattern.subn(
        cell_magic_replacer, formatted_code
    )

    # Handle %sql (line magic)
    line_magic_pattern = re.compile(
        r"""
        ^(?P<indent>^[ \t]*)               # Indentation
        %(?P<magic>sql)\b[ \t]+            # %sql magic command
        (?P<sql>.+)                        # SQL code
        """,
        re.VERBOSE | re.MULTILINE,
    )

    def line_magic_replacer(match):
        nonlocal changed
        indent = match.group("indent")
        magic = match.group("magic")
        sql_code = match.group("sql")

        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return original

        # Format the SQL code
        formatted_sql = format_sql_code(sql_code)

        # Remove leading indentation from each line
        formatted_sql = "\n".join(line.lstrip() for line in formatted_sql.split("\n"))

        # Reconstruct the magic command
        new_magic = f"{indent}%{magic} {formatted_sql.replace('\n', ' ')}"

        logger.debug("Formatting %sql magic command.")
        logger.debug(f"Original SQL: {sql_code}")
        logger.debug(f"Formatted SQL: {formatted_sql}")

        changed = True
        return new_magic

    # Replace all %sql magic commands with formatted SQL
    formatted_code, num_subs_line_magic = line_magic_pattern.subn(
        line_magic_replacer, formatted_code
    )

    return formatted_code, changed


def format_sql_in_notebook(notebook_path, sql_keywords):
    """Processes a Jupyter notebook to format SQL code within it."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        notebook_changed = False

        for idx, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                original_code = cell.source
                logger.debug(f"Processing cell {idx + 1}:\n{original_code}")

                # Format SQL in the code cell
                formatted_code, changed = format_sql_in_code_cell(
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
        if format_sql_in_notebook(notebook_path, sql_keywords):
            any_notebook_changed = True

    if any_notebook_changed:
        logger.info("SQL formatting changes were made.")
        sys.exit(1)  # Indicate that changes were made
    else:
        logger.info("No SQL formatting changes were necessary.")
        sys.exit(0)  # Indicate that no changes were made


if __name__ == "__main__":
    main()
