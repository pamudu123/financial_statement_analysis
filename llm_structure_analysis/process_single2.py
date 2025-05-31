# process_sequentially.py
import ollama
import fitz  # PyMuPDF
import os
import json

# Import from common_utils
from llm_utils import (
    OLLAMA_MULTIMODAL_MODEL,
    OLLAMA_HOST,
    convert_pdf_page_to_image_base64,
    print_llm_metrics,
    strip_json_markdown
)

def check_single_image_for_tables_ollama(model_name, image_base64, ollama_host_url, page_number_display):
    """
    Sends a single page image to Ollama and asks if it contains a financial data table.
    Expects a JSON object response from the LLM.
    Returns "YES", "NO", or an "ERROR_*" string.
    """
    if not image_base64:
        return "ERROR_NO_IMAGE"

    prompt = f"""
<Instructions>
You are an AI assistant specialized in document analysis. Your task is to determine if the provided page image from a PDF document contains one or more **financial data tables**, which may sometimes span multiple pages or have direct visual links/references to related notes (potentially on the same or nearby pages).
The documents are typically financial reports, including statements like Profit or Loss (P&L), Balance Sheets, Cash Flow Statements, and Notes to Financial Statements.

You will receive a single page image.
The actual page number from the PDF document for this page is: {page_number_display}.
You must use this page number as 'page_number' in your JSON response.

Definition of a Financial Data Table (including multi-page and linked notes considerations):
A financial data table is a structured presentation of financial information, typically in rows and columns, often displaying monetary values and comparisons across different periods or categories. Key characteristics include:
- A grid-like layout of financial data, with clear rows representing financial items and columns often representing time periods, currencies, or other categories.
- Presence of numerical data that represents financial figures (e.g., amounts in various currencies, possibly with scaling like '000s or Millions indicated). Numbers within columns are usually aligned.
- Column headers that clearly label the financial data below (e.g., "Note", "Description", "2024 $", "2023 $"). For tables spanning multiple pages, these headers might appear only on the initial page or be implied by a strong continuation of the columnar data structure.
- Row labels that identify specific financial elements, accounts, or metrics (e.g., "Revenue", "Cost of Goods Sold", "Property, Plant, and Equipment").
- May include visual cues like lines separating rows and columns, or rely on consistent spacing and alignment to create the tabular structure. Subtotals and totals are commonly present.
- Sometimes, a financial table on one page might have direct visual links (e.g., lines, arrows) or clear textual references (e.g., "(see Note X)") to related financial data or explanations that might be presented in a less structured format or in another table/list nearby (even on a different page). Your determination of a "financial data table" on a given page should consider if the primary content of that page is the structured financial data itself, even if it has links to supporting details elsewhere. A page that ONLY contains the supporting note without a clear table of financial figures on it would typically be "NO". However, if a page contains a significant portion of a financial data table AND has such links/references, it should be considered "YES".

Common examples include:
- Statements of financial performance (like the Profit and Loss statement).
- Statements of financial position (like the Balance Sheet).
- Statements of cash flows.
- Notes to the financial statements that present detailed breakdowns or movements in financial accounts in a tabular format.

Exclude:
- Pages with only narrative text, even if they discuss financial matters.
- Tables of contents.
- Simple lists of items that do not primarily present financial data in a comparative or structured numerical way.
- Organizational charts or similar non-financial diagrams.

Focus on whether the page primarily presents financial information in a structured, row-and-column format with numerical data. Consider the presence of headers, clear financial line items, and the overall organization of data. Even if a full table extends beyond one page, each page that contains a discernible part of that financial data table (with headers or a clear continuation of the structure) should be identified as "YES". Pages with direct links or references to such tables should also be considered in context.
</Instructions>

<Examples_Provided_By_User>
The user has provided an example showing a "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME" which is a financial table. The prompt should recognize pages containing such structured financial statements (even if they might refer to notes elsewhere).
User also provided examples of:
- A "STATEMENT OF CASH FLOWS" showing items like "Receipts from customers", "Payments to suppliers", "Interest paid" with corresponding dollar amounts for two years (e.g., 2023 $, 2022 $).
- A "NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS" section, for example, detailing "Movement in temporary differences" under "INCOME TAX", showing items like "Opening balance", "Adjustments", "Recognised in income", "Closing balance" with dollar amounts ($Million) for two years.
- A "RELATED PARTY TRANSACTIONS" note showing tabular data for items like "Management fee from immediate parent entity", "Information system services", "Disposal of non-current asset" and a separate table for "Loan/payables to related parties" with corresponding dollar amounts for two years.
</Examples_Provided_By_User>

<General_Examples_Format_Focus>
    <Example1 - Page with a financial statement table (P&L, Balance Sheet, etc.)>
    Output:
    {{"page_number": "{page_number_display}", "has_table": "YES"}}
    </Example1>

    <Example2 - Page with only text or non-financial content>
    Output:
    {{"page_number": "{page_number_display}", "has_table": "NO"}}
    </Example2>
</General_Examples_Format_Focus>

---
Now, for the current page image you have received:

Your response MUST be a valid JSON object.
The object must have two keys:
1. "page_number": The actual page number from the PDF document (this must be the number: {page_number_display} that you were given for this page).
2. "has_table": A string value, either "YES" if a financial data table (considering multi-page and linked notes aspects) is the primary content of the page, or "NO" if not.

Ensure your entire response is ONLY this JSON object, with no other text before or after it.
Process the image and provide a single JSON object as the response.
"""
    print(f"\nSending Page {page_number_display} to Ollama model '{model_name}' for financial table detection...")

    try:
        client = ollama.Client(host=ollama_host_url)
        response_data = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_base64] # Ollama client expects a list of images
                }
            ],
            options={'temperature': 0.0} # For more deterministic output
        )

        print_llm_metrics(response_data, f"Page {page_number_display}")

        llm_output_str = response_data['message']['content'].strip()
        print(f"Ollama raw response for Page {page_number_display}: '{llm_output_str}'")

        llm_output_str = strip_json_markdown(llm_output_str)
        print(f'Ollama processed response : llm_output_str = {llm_output_str}')
        print("--"* 25)

        try:
            llm_json_response = json.loads(llm_output_str)

            if not isinstance(llm_json_response, dict):
                print(f"Error: LLM JSON response is not a dictionary for Page {page_number_display}.")
                return "ERROR_FORMAT"

            if "page_number" in llm_json_response and "has_table" in llm_json_response:
                try:
                    resp_page_num_str = str(llm_json_response["page_number"])
                    resp_page_num_int = int(resp_page_num_str)
                    has_table_val = str(llm_json_response["has_table"]).upper()

                    if resp_page_num_int != page_number_display:
                        print(f"Warning: LLM returned page_number {resp_page_num_int} which does not match the expected page number {page_number_display}.")
                        # Depending on strictness, you might return an error here or proceed
                        # For now, we'll trust the has_table value but log a warning.

                    if has_table_val == "YES" or has_table_val == "NO":
                        return has_table_val
                    else:
                        print(f"Warning: Invalid 'has_table' value '{llm_json_response['has_table']}' for page {page_number_display}.")
                        return "ERROR_VALUE"
                except ValueError:
                    print(f"Warning: LLM returned non-integer page_number '{llm_json_response['page_number']}' for page {page_number_display}.")
                    return "ERROR_PAGE_NUM_TYPE"
            else:
                print(f"Warning: Invalid item structure in LLM JSON response for Page {page_number_display}: {llm_json_response}")
                return "ERROR_STRUCTURE"

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON response from LLM for Page {page_number_display}.")
            return "ERROR_JSON_DECODE"

    except ollama.ResponseError as e:
        print(f"Ollama API Error for Page {page_number_display}: {e}")
        return "ERROR_OLLAMA_API"
    except Exception as e:
        print(f"Unexpected error during Ollama call for Page {page_number_display}: {e}")
        return "ERROR_UNEXPECTED_CALL"


def detect_tables_in_pdf_sequentially(pdf_path,
                                     model_name_param, ollama_host_param,
                                     image_dpi=150, total_pages_to_process_param=None):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return {}

    print(f"Starting sequential financial table detection for PDF: {pdf_path}")
    print(f"Ollama model: {model_name_param}")
    print(f"Image DPI: {image_dpi}")
    print("-" * 30)

    table_detection_results = {}
    doc = None

    try:
        doc = fitz.open(pdf_path)
        actual_total_pages_in_pdf = doc.page_count

        if actual_total_pages_in_pdf == 0:
            print("PDF is empty or unreadable. Aborting.")
            return {}
        print(f"Total pages available in PDF: {actual_total_pages_in_pdf}")

    except Exception as e:
        print(f"Error opening PDF with PyMuPDF: {e}")
        if doc: doc.close()
        return {}

    if total_pages_to_process_param is None:
        pages_to_process_count = actual_total_pages_in_pdf
        print(f"No specific page limit provided. Processing all {pages_to_process_count} pages.")
    else:
        pages_to_process_count = min(total_pages_to_process_param, actual_total_pages_in_pdf)
        if pages_to_process_count == total_pages_to_process_param:
            print(f"Processing {pages_to_process_count} pages as specified.")
        else:
            print(f"Warning: Requested {total_pages_to_process_param} pages, but PDF only has {actual_total_pages_in_pdf}. Processing {pages_to_process_count} pages.")

        if pages_to_process_count == 0:
            print("No pages to process.")
            if doc: doc.close()
            return {}

    for current_page_index in range(pages_to_process_count):
        page_num_internal = current_page_index  # 0-based for PyMuPDF
        page_num_display = current_page_index + 1  # 1-based for user/LLM

        print(f"Preparing Page {page_num_display} for processing...")
        img_base64 = convert_pdf_page_to_image_base64(doc, page_num_internal, dpi=image_dpi)

        if img_base64:
            result_status = check_single_image_for_tables_ollama(
                model_name_param,
                img_base64,
                ollama_host_param,
                page_num_display
            )
            table_detection_results[page_num_display] = result_status
            print(f"**** LLM Result for Page {page_num_display}: {result_status} ****")
        else:
            print(f"Failed to convert Page {page_num_display} to image. Marking as ERROR_CONVERSION.")
            table_detection_results[page_num_display] = "ERROR_CONVERSION"
        
        print("-" * 20)

    if doc:
        doc.close()

    print("\n" + "=" * 30)
    print("Sequential Financial Table Detection Complete.")
    print("=" * 30)
    return table_detection_results

if __name__ == "__main__":
    PDF_FILE_PATH = "docs/YHI PTY LTD.pdf"  # Replace with your PDF file path
    TOTAL_PAGES_TO_PROCESS = None # Process all pages, or set a number e.g. 5
    IMAGE_CONVERSION_DPI = 150 # Lower DPI for potentially faster processing and smaller payload

    # Ensure the 'docs' directory exists if a path is specified like 'docs/filename.pdf'
    pdf_dir = os.path.dirname(PDF_FILE_PATH)
    if pdf_dir and not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"Created directory: {pdf_dir}")
        print(f"Please place your PDF file '{os.path.basename(PDF_FILE_PATH)}' in the '{pdf_dir}' directory to run this example.")


    if not os.path.exists(PDF_FILE_PATH):
        print(f"FATAL ERROR: PDF file does not exist at '{PDF_FILE_PATH}'. Please check the path.")
    else:
        results = detect_tables_in_pdf_sequentially(
            pdf_path=PDF_FILE_PATH,
            model_name_param=OLLAMA_MULTIMODAL_MODEL, # from common_utils
            ollama_host_param=OLLAMA_HOST,           # from common_utils
            image_dpi=IMAGE_CONVERSION_DPI,
            total_pages_to_process_param=TOTAL_PAGES_TO_PROCESS
        )
        print("\nFinal Sequential Financial Table Detection Results:")
        for page_num in sorted(results.keys()):
            print(f"  Page {page_num}: {results[page_num]}")