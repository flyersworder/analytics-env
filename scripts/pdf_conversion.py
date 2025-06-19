from pathlib import Path
import json

# docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# other pdf libraries
import pdfplumber
from pymupdf4llm import to_markdown
import fitz  # PyMuPDF

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.parent
BASE_OUTPUT_DIR = PROJECT_ROOT / "conversion_outputs"


# --- Helper Functions ---
def _serialize_and_merge_rows(raw_table):
    """Converts a raw table (list of lists) into a list of merged dictionaries."""
    if not raw_table or len(raw_table) < 1:
        return []

    # Clean header and data rows
    header = [
        str(h).replace("\n", " ").strip() if h else f"column_{i+1}"
        for i, h in enumerate(raw_table[0])
    ]
    data_rows = [
        [
            str(cell).replace("\n", " ").strip() if cell is not None else ""
            for cell in row
        ]
        for row in raw_table[1:]
    ]

    merged_rows = []
    for row_list in data_rows:
        if row_list[0]:  # A new parameter entry starts if the first cell is not empty.
            merged_rows.append(dict(zip(header, row_list)))
        elif merged_rows:  # Otherwise, it's a continuation of the previous parameter.
            last_row_dict = merged_rows[-1]
            continuation_dict = dict(zip(header, row_list))
            for key, value in continuation_dict.items():
                if value:  # Only append if there's new information.
                    last_row_dict[key] = (
                        f"{last_row_dict.get(key, '')}\n{value}".strip()
                    )
    return merged_rows


# --- Conversion Functions ---
def convert_with_docling(pdf_path, markdown_output_path, tables_output_path):
    """Converts PDF to markdown and extracts tables using docling with detailed options."""
    print(f"Processing {pdf_path.name} with docling...")
    try:
        # Configure detailed table structure options
        pdf_opts = PdfPipelineOptions(do_table_structure=True)
        pdf_opts.table_structure_options.do_cell_matching = True
        pdf_opts.table_structure_options.mode = TableFormerMode.FAST
        pdf_opts.do_ocr = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts, backend=PyPdfiumDocumentBackend
                ),
            }
        )
        conv_res = converter.convert(str(pdf_path))
        doc = conv_res.document

        # Save markdown
        with open(markdown_output_path, "w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown())
        print(f"docling markdown saved to {markdown_output_path}")

        # Debug: Check what's available in the document
        print(f"Document has {len(doc.tables)} tables")

        # Extract, serialize, and save tables using the conversion result
        all_tables_structured = {}
        for table_ix, table in enumerate(conv_res.document.tables):
            try:
                table_df = table.export_to_dataframe()
                print(
                    f"Table {table_ix}: {table_df.shape[0]} rows, {table_df.shape[1]} columns"
                )

                # For now, assign all tables to page 1 - we'll improve this later
                page_key = "page_1"
                if page_key not in all_tables_structured:
                    all_tables_structured[page_key] = []

                raw_table = [table_df.columns.tolist()] + table_df.values.tolist()
                serialized_table = _serialize_and_merge_rows(raw_table)
                if serialized_table:
                    all_tables_structured[page_key].append(serialized_table)

            except Exception as e:
                print(f"Could not process table {table_ix} with docling: {e}")

        if all_tables_structured:
            with open(tables_output_path, "w", encoding="utf-8") as f:
                json.dump(all_tables_structured, f, indent=4)
            print(f"docling tables saved to {tables_output_path}")
        else:
            print(f"No tables found by docling in {pdf_path.name}")

    except Exception as e:
        print(f"Error converting {pdf_path.name} with docling: {e}")


def convert_with_pdfplumber(pdf_path, text_output_path, tables_output_path):
    """Converts PDF to text and extracts tables using pdfplumber."""
    print(f"Processing {pdf_path.name} with pdfplumber...")
    all_tables_structured = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            for page_num, page in enumerate(pdf.pages):
                text_content += f"--- Page {page_num + 1} ---\n{page.extract_text(x_tolerance=1) or ''}\n\n"
                tables = page.extract_tables()
                if tables:
                    page_key = f"page_{page_num + 1}"
                    all_tables_structured[page_key] = [
                        _serialize_and_merge_rows(t) for t in tables if t
                    ]

        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"pdfplumber text saved to {text_output_path}")

        if all_tables_structured:
            with open(tables_output_path, "w", encoding="utf-8") as f:
                json.dump(all_tables_structured, f, indent=4)
            print(f"pdfplumber tables saved to {tables_output_path}")
        else:
            print(f"No tables found by pdfplumber in {pdf_path.name}")

    except Exception as e:
        print(f"Error converting {pdf_path.name} with pdfplumber: {e}")


def convert_with_pymupdf(pdf_path, tables_output_path):
    """Extracts tables using PyMuPDF (fitz)."""
    print(f"Extracting tables from {pdf_path.name} with PyMuPDF...")
    all_tables_structured = {}
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                tables = page.find_tables()
                if tables:
                    page_key = f"page_{page_num + 1}"
                    extracted_tables = [t.extract() for t in tables]
                    all_tables_structured[page_key] = [
                        _serialize_and_merge_rows(t) for t in extracted_tables if t
                    ]

        if all_tables_structured:
            with open(tables_output_path, "w", encoding="utf-8") as f:
                json.dump(all_tables_structured, f, indent=4)
            print(f"PyMuPDF tables saved to {tables_output_path}")
        else:
            print(f"No tables found by PyMuPDF in {pdf_path.name}")

    except Exception as e:
        print(f"Error extracting tables from {pdf_path.name} with PyMuPDF: {e}")


def convert_with_pymupdf4llm(pdf_path, markdown_output_path):
    """Converts PDF to markdown using pymupdf4llm."""
    print(f"Converting {pdf_path.name} to markdown with pymupdf4llm...")
    try:
        with open(markdown_output_path, "w", encoding="utf-8") as f:
            f.write(to_markdown(str(pdf_path)))
        print(f"pymupdf4llm markdown saved to {markdown_output_path}")
    except Exception as e:
        print(f"Error converting {pdf_path.name} with pymupdf4llm: {e}")


def main():
    """Main function to run all conversions."""
    BASE_OUTPUT_DIR.mkdir(exist_ok=True)
    pdf_files = list(PROJECT_ROOT.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {PROJECT_ROOT}\n")

    for pdf_path in pdf_files:
        pdf_stem = pdf_path.stem
        output_dir = BASE_OUTPUT_DIR / pdf_stem
        output_dir.mkdir(exist_ok=True)
        print(f"--- Processing: {pdf_path.name} -> Output Dir: {output_dir} ---")

        # Define output paths
        docling_md = output_dir / f"{pdf_stem}_docling.md"
        docling_tables = output_dir / f"{pdf_stem}_docling_tables.json"
        pdfplumber_text = output_dir / f"{pdf_stem}_pdfplumber.md"
        pdfplumber_tables = output_dir / f"{pdf_stem}_pdfplumber_tables.json"
        pymupdf_tables = output_dir / f"{pdf_stem}_pymupdf_tables.json"
        pymupdf4llm_md = output_dir / f"{pdf_stem}_pymupdf4llm.md"

        # Run conversions
        convert_with_docling(pdf_path, docling_md, docling_tables)
        convert_with_pdfplumber(pdf_path, pdfplumber_text, pdfplumber_tables)
        convert_with_pymupdf(pdf_path, pymupdf_tables)
        convert_with_pymupdf4llm(pdf_path, pymupdf4llm_md)
        print(f"--- Finished: {pdf_path.name} ---\n")

    print("All conversions complete.")


if __name__ == "__main__":
    main()
