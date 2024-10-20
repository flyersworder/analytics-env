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


def format_sql_in_code_cell(code, sql_keywords):
    """Formats SQL strings assigned to variables in a code cell."""
    # Regular expression to match variable assignments with strings
    pattern = re.compile(
        r"""
        (?P<indent>^[ \t]*)                # Indentation at the start of the line
        (?P<var_name>\w+)                  # Variable name
        [ \t]*=[ \t]*                      # Assignment operator
        (?P<quote_prefix>[frbuFRBU]*)      # Optional prefixes (f, r, b, u)
        (?P<quote_char>['"]{1,3})          # Opening quote(s)
        (?P<sql_code>.*?)
        (?P=quote_char)                    # Closing quote(s)
        """,
        re.VERBOSE | re.DOTALL | re.MULTILINE,
    )

    def replacer(match):
        indent = match.group("indent")
        var_name = match.group("var_name")
        quote_prefix = match.group("quote_prefix")
        quote_char = match.group("quote_char")
        sql_code = match.group("sql_code")

        # Check if the string contains SQL keywords
        if not contains_sql_keywords(sql_code, sql_keywords):
            return match.group(0)  # Return the original string

        # Remove leading and trailing whitespace
        sql_code = sql_code.strip()

        # Handle f-strings
        is_f_string = "f" in quote_prefix.lower()

        # If it's an f-string, preserve placeholders
        if is_f_string:
            # Find placeholders in the f-string
            placeholder_pattern = re.compile(r"(\{[^}]*\})")
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
        formatted_sql = f"{triple_quote_char}{formatted_sql}{triple_quote_char}"

        # Reconstruct the assignment
        new_quote_prefix = quote_prefix
        new_line = f"{indent}{var_name} = {new_quote_prefix}{formatted_sql}"

        return new_line

    # Replace all matches in the code
    new_code = pattern.sub(replacer, code)

    return new_code


def format_sql_in_notebook(notebook_path, sql_keywords):
    """Processes a Jupyter notebook to format SQL code within it."""
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        changed = False
        for cell in nb.cells:
            if cell.cell_type == "code":
                original_code = cell.source
                logger.debug(f"Processing cell:\n{original_code}")

                # Format SQL strings assigned to variables
                formatted_code = format_sql_in_code_cell(original_code, sql_keywords)

                # Determine if any changes were made
                code_changed = formatted_code != original_code

                if code_changed:
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
                    logger.debug("No changes detected in code cell.")
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
    sql_keywords = load_config(args.config)

    any_notebook_changed = False
    for notebook_path in args.notebooks:
        logger.info(f"Processing notebook: {notebook_path}")
        if format_sql_in_notebook(
            notebook_path,
            sql_keywords,
        ):
            any_notebook_changed = True

    if any_notebook_changed:
        logger.info("SQL formatting changes were made.")
        sys.exit(1)  # Indicate that changes were made
    else:
        logger.info("No SQL formatting changes were necessary.")
        sys.exit(0)  # Indicate that no changes were made
