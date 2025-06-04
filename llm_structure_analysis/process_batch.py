# process_batch.py
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

def check_image_batch_for_tables_ollama(model_name, image_batch_base64, ollama_host_url, batch_page_numbers_display):
    """
    Sends a batch of page images to Ollama and asks if each contains a financial data table.
    Expects a JSON response from the LLM.
    Returns a list of "YES", "NO", or "ERROR" strings, corresponding to the input batch.
    """
    if not image_batch_base64:
        return ["ERROR_NO_IMAGES"] * len(batch_page_numbers_display) if batch_page_numbers_display else []

    num_images_in_batch = len(image_batch_base64)
    actual_page_numbers_string = ", ".join(map(str, batch_page_numbers_display))

    prompt = f"""
<Instructions>
You are an AI assistant specialized in document analysis. Your task is to determine if each provided page image from a PDF document contains one or more **financial data tables**, which may sometimes span multiple pages or have direct visual links/references to related notes (potentially on the same or nearby pages).
The documents are typically financial reports, including statements like Profit or Loss (P&L), Balance Sheets, Cash Flow Statements, and Notes to Financial Statements.

You will receive a batch of {num_images_in_batch} page images.
The actual page numbers from the PDF document for this batch are: {actual_page_numbers_string}.
For example, if the page numbers are "4, 5, 6", these are the numbers you must use as 'page_number' in your JSON response.

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

<Examples>

    <Example1>
    Assume num_images_in_batch = 2
    Assume actual_page_numbers_string = "2, 11"

    Input Image 1 Content (Actual Page 2):
    **Statement of Profit or Loss and Other Comprehensive Income**
    For the Year Ended ...
                            Note     2024 $       2023 $
    Revenue                   3   38,403,987   35,363,211
    Other income              4      383,018      314,453
    ... (table continues) ...

    Input Image 2 Content (Actual Page 11):
    **Chairman's Report**
    The 2024 financial year presented a dynamic environment for our operations. We navigated fluctuating market conditions and inflationary pressures... (continues with paragraphs of text).

    Expected Output for this batch:
    [
      {{"page_number": "2", "has_table": "YES"}},
      {{"page_number": "11", "has_table": "NO"}}
    ]
    </Example1>

    <Example2>
    Assume num_images_in_batch = 2
    Assume actual_page_numbers_string = "6, 15"

    Input Image 1 Content (Actual Page 6 - Continuation of a larger table, headers on a previous page):
    (Amounts in $ thousands)
    Consultancy fees                                  85.6       75.3
    Legal and professional fees                      120.2      110.9
    Marketing and advertising                         95.0       90.1
    ... (table continues with clear columnar data) ...

    Input Image 2 Content (Actual Page 15 - A note with a clear financial data table):
    **Note 12: Related Party Transactions**
    Transactions between related parties are on normal commercial terms and conditions no more favourable than those available to other parties unless otherwise stated.
    The following transactions occurred with related parties:
                                            2024 ($)     2023 ($)
    Sales of goods to associate A           150,000      120,000
    Management fees from parent entity       50,000       45,000
    Purchases from entity B                  75,000       60,000

    Expected Output for this batch:
    [
      {{"page_number": "6", "has_table": "YES"}},
      {{"page_number": "15", "has_table": "YES"}}
    ]
    </Example2>

    <Example3>
    Assume num_images_in_batch = 3
    Assume actual_page_numbers_string = "3, 7, 8"

    Input Image 1 Content (Actual Page 3 - Table of Contents):
    **Contents**
    Directors' Report ....................................... 1
    Auditor's Independence Declaration ...................... 3
    Financial Statements .................................... 4
    Statement of Profit or Loss ............................. 5

    Input Image 2 Content (Actual Page 7 - Detailed financial table):
    **Notes to the Consolidated Financial Statements**
    **Note 5. Income Tax**
    (a) Income tax expense
                                            Group
                                        2024 $'000   2023 $'000
    Current tax expense                  1,200        1,100
    Deferred tax expense related to
      origination and reversal of
      temporary differences                (50)          75
    Total income tax expense             1,150        1,175

    Input Image 3 Content (Actual Page 8 - Text-heavy page with no primary table):
    **Auditor's Opinion**
    In our opinion, the accompanying financial report gives a true and fair view of the financial position of the Company as at 31 December 2024... (continues with audit opinion text).

    Expected Output for this batch:
    [
      {{"page_number": "3", "has_table": "NO"}},
      {{"page_number": "7", "has_table": "YES"}},
      {{"page_number": "8", "has_table": "NO"}}
    ]
    </Example3>

</Examples>

---
Now, for the current batch of images you have received:

Your response MUST be a valid JSON array where each element is an object.
Each object in the array must correspond to one of the provided page images in this current batch.
Each object must have two keys:
1. "page_number": The actual page number from the PDF document (this must be one of the numbers from the list: {actual_page_numbers_string} that you were given for this batch). Ensure this is a string in the output if the input {actual_page_numbers_string} implies string page numbers, or an integer if they are integers. The examples use strings.
2. "has_table": A string value, either "YES" if a financial data table (considering multi-page and linked notes aspects) is the primary content of the page, or "NO" if not.
The number of items in the JSON array must match the number of images you received in this batch.

Ensure your entire response is ONLY this JSON array, with no other text before or after it.
There are {num_images_in_batch} images in this current batch. Process all of them and provide a JSON object for each.
"""
    print(f"\nSending batch of {num_images_in_batch} images (Pages: {actual_page_numbers_string}) to Ollama model '{model_name}' for financial table detection...")

    try:
        client = ollama.Client(host=ollama_host_url)
        response_data = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': image_batch_base64
                }
            ],
            options={'temperature': 0.0} # For more deterministic output
        )

        print_llm_metrics(response_data, f"Pages {actual_page_numbers_string}")

        llm_output_str = response_data['message']['content'].strip()
        print(f"Ollama raw response for batch (Pages {actual_page_numbers_string}): '{llm_output_str}'")

        llm_output_str = strip_json_markdown(llm_output_str) # Use common util
        print(f'Ollama processed response : llm_output_str = {llm_output_str}')
        print("--"* 25)

        parsed_results = []
        try:
            llm_json_response = json.loads(llm_output_str)

            if not isinstance(llm_json_response, list) or len(llm_json_response) != num_images_in_batch:
                print(f"Error: LLM JSON response is not a list of the correct length for batch (Pages {actual_page_numbers_string}). Expected {num_images_in_batch} items, got {len(llm_json_response) if isinstance(llm_json_response, list) else 'not a list'}.")
                return ["ERROR_FORMAT"] * num_images_in_batch

            page_to_index_map = {page_num: i for i, page_num in enumerate(batch_page_numbers_display)}
            temp_batch_results = ["ERROR_PARSE"] * num_images_in_batch

            for item in llm_json_response:
                if isinstance(item, dict) and "page_number" in item and "has_table" in item:
                    try:
                        resp_page_num_str = str(item["page_number"]) # Keep as string for dict lookup if page_numbers are strings
                        resp_page_num_int = int(resp_page_num_str)   # Convert to int for comparison/use
                        has_table_val = str(item["has_table"]).upper()

                        # Check if the page number from LLM response is in our original batch list
                        if resp_page_num_int in page_to_index_map:
                            item_idx = page_to_index_map[resp_page_num_int]
                            if has_table_val == "YES" or has_table_val == "NO":
                                temp_batch_results[item_idx] = has_table_val
                            else:
                                temp_batch_results[item_idx] = "ERROR_VALUE"
                                print(f"Warning: Invalid 'has_table' value '{item['has_table']}' for page {resp_page_num_int} in batch (Pages {actual_page_numbers_string}).")
                        else:
                            print(f"Warning: LLM returned page_number {resp_page_num_int} which was not in the expected batch page numbers: {batch_page_numbers_display}.")
                    except ValueError:
                        print(f"Warning: LLM returned non-integer page_number '{item['page_number']}' in batch (Pages {actual_page_numbers_string}).")
                else:
                    print(f"Warning: Invalid item structure in LLM JSON response for batch (Pages {actual_page_numbers_string}): {item}")

            if any(res == "ERROR_PARSE" for res in temp_batch_results):
                 print(f"Warning: Not all items in LLM JSON response were valid or correctly structured for batch (Pages {actual_page_numbers_string}). Some results may be 'ERROR_PARSE'.")
            parsed_results = temp_batch_results

        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON response from LLM for batch (Pages {actual_page_numbers_string}).")
            parsed_results = ["ERROR_JSON_DECODE"] * num_images_in_batch

        return parsed_results

    except ollama.ResponseError as e:
        print(f"Ollama API Error for batch (Pages {actual_page_numbers_string}): {e}")
        return ["ERROR_OLLAMA_API"] * num_images_in_batch
    except Exception as e:
        print(f"Unexpected error during Ollama call for batch (Pages {actual_page_numbers_string}): {e}")
        return ["ERROR_UNEXPECTED_CALL"] * num_images_in_batch


def detect_tables_in_pdf_page_batches(pdf_path, pages_per_llm_call,
                                      model_name_param, ollama_host_param,
                                      image_dpi=150, total_pages_to_process_param=None):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return {}

    print(f"Starting batched financial table detection for PDF: {pdf_path}")
    print(f"Pages per LLM call (batch size): {pages_per_llm_call}")
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

    current_page_index = 0
    while current_page_index < pages_to_process_count:
        batch_image_data = []
        batch_original_page_numbers = [] # Display numbers (1-based)

        for _ in range(pages_per_llm_call):
            if current_page_index < pages_to_process_count:
                page_num_internal = current_page_index # 0-based for PyMuPDF
                page_num_display = current_page_index + 1 # 1-based for user/LLM

                print(f"Preparing Page {page_num_display} for batch...")
                img_base64 = convert_pdf_page_to_image_base64(doc, page_num_internal, dpi=image_dpi)

                if img_base64:
                    batch_image_data.append(img_base64)
                    batch_original_page_numbers.append(page_num_display)
                else:
                    print(f"Failed to convert Page {page_num_display} to image. Marking as ERROR_CONVERSION.")
                    table_detection_results[page_num_display] = "ERROR_CONVERSION"

                current_page_index += 1
            else:
                break # No more pages to process in the PDF

        if not batch_image_data: # All pages in this potential batch failed conversion or no pages left
            if current_page_index >= pages_to_process_count:
                break # All pages processed or attempted
            else:
                continue # Some pages failed conversion, loop to next potential batch start

        batch_results_from_llm = check_image_batch_for_tables_ollama(
            model_name_param,
            batch_image_data,
            ollama_host_param,
            batch_original_page_numbers # These are the display numbers
        )
        print(f"**** LLM Results for Batch (Pages {batch_original_page_numbers}): {batch_results_from_llm} ****")

        for idx_in_batch, result_status in enumerate(batch_results_from_llm):
            if idx_in_batch < len(batch_original_page_numbers):
                original_page_num = batch_original_page_numbers[idx_in_batch]
                if table_detection_results.get(original_page_num) != "ERROR_CONVERSION" or \
                   result_status not in ["ERROR_NO_IMAGES"]:
                    table_detection_results[original_page_num] = result_status
                print(f"Page {original_page_num}: Financial table detected = {table_detection_results[original_page_num]} (from batch)")
            else:
                print(f"Warning: Result status '{result_status}' received for index {idx_in_batch} which is out of bounds for current batch page numbers.")
        print("-" * 20)

    if doc:
        doc.close()

    print("\n" + "=" * 30)
    print("Batched Financial Table Detection Complete.")
    print("=" * 30)
    return table_detection_results

if __name__ == "__main__":
    PDF_FILE_PATH = "docs/YHIPTYLTD.pdf"
    TOTAL_PAGES_TO_PROCESS = None       # Process all pages
    PAGES_PER_LLM_CALL = 2              # Number of pages to process in one LLM call
    IMAGE_CONVERSION_DPI = 150          # Lower DPI for potentially faster processing and smaller payload

    if not os.path.exists(PDF_FILE_PATH):
        print(f"FATAL ERROR: PDF file does not exist at '{PDF_FILE_PATH}'. Please check the path.")
    else:
        if not os.path.exists(os.path.dirname(PDF_FILE_PATH)) and os.path.dirname(PDF_FILE_PATH):
             os.makedirs(os.path.dirname(PDF_FILE_PATH), exist_ok=True)
             print(f"Created directory: {os.path.dirname(PDF_FILE_PATH)}")
             print(f"Please place your PDF file '{os.path.basename(PDF_FILE_PATH)}' in the 'docs' directory to run this example.")

        results = detect_tables_in_pdf_page_batches(
            pdf_path=PDF_FILE_PATH,
            pages_per_llm_call=PAGES_PER_LLM_CALL,
            model_name_param=OLLAMA_MULTIMODAL_MODEL, # from llm_utils
            ollama_host_param=OLLAMA_HOST,            # from llm_utils
            image_dpi=IMAGE_CONVERSION_DPI,
            total_pages_to_process_param=TOTAL_PAGES_TO_PROCESS
        )
        print("\nFinal Batched Financial Table Detection Results:")
        for page_num in sorted(results.keys()):
            print(f"  Page {page_num}: {results[page_num]}")