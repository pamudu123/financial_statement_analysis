import ollama
import fitz  # PyMuPDF
import os
import json
from enum import Enum

class ResultKeys(Enum):
    PAGE_NUMBER = "page_number"
    STATEMENT_TITLE = "statement_title"
    REPORTING_PERIOD = "reporting_period_info"
    CURRENCY_SYMBOL = "currency_symbol"
    ROUNDING_SCALE = "rounding_scale"
    COLUMN_HEADERS = "column_headers"
    LINE_ITEMS = "line_items"
    # Line item keys
    DESCRIPTION = "description"
    NOTE_REFERENCE = "note_reference"
    VALUES = "values"
    IS_SUBTOTAL = "is_subtotal"
    IS_TOTAL = "is_total"
    INDENTATION_LEVEL = "indentation_level"

from llm_utils import (
    OLLAMA_MULTIMODAL_MODEL,
    OLLAMA_HOST,
    convert_pdf_page_to_image_base64,
    print_llm_metrics,
    strip_json_markdown)


# --- Stage 1: Check if page contains a financial table ---
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
You are an AI assistant specialized in document analysis. Your task is to determine if the provided page image from a PDF document contains one or more **financial data tables**.
You will receive a single page image. The actual page number from the PDF document for this page is: {page_number_display}.
Definition of a Financial Data Table:
A structured presentation of financial information in rows and columns, often displaying monetary values. Key characteristics include:
- Grid-like layout with financial data.
- Numerical data representing financial figures.
- Column headers and row labels.
- May include visual cues like lines or rely on spacing/alignment.
Common examples: Statements of financial performance (P&L), financial position (Balance Sheet), cash flows, and tabular notes.
Exclude: Narrative text, tables of contents, non-financial diagrams.

Focus on whether the page primarily presents financial information in a structured, row-and-column format with numerical data.
</Instructions>

<OutputFormat>
Your response MUST be a valid JSON object with two keys:
1. "page_number": The actual page number (must be {page_number_display}).
2. "has_table": A string value, either "YES" or "NO".
Example: {{"page_number": "{page_number_display}", "has_table": "YES"}}
Ensure your entire response is ONLY this JSON object.
</OutputFormat>
---
Process the image for page {page_number_display} and provide the JSON object.
"""
    print(f"\n[Stage 1] Sending Page {page_number_display} to Ollama model '{model_name}' for table detection...")

    try:
        client = ollama.Client(host=ollama_host_url)
        response_data = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_base64]}],
            options={'temperature': 0.0}
        )
        print_llm_metrics(response_data, f"Page {page_number_display} (Table Detection)")
        llm_output_str = strip_json_markdown(response_data['message']['content'])
        print(f"[Stage 1] Ollama processed response for Page {page_number_display}: '{llm_output_str}'")

        llm_json_response = json.loads(llm_output_str)
        if not isinstance(llm_json_response, dict): return "ERROR_FORMAT"
        
        resp_page_num = int(str(llm_json_response.get(ResultKeys.PAGE_NUMBER.value)))
        has_table_val = str(llm_json_response.get("has_table")).upper()

        if resp_page_num != page_number_display:
            print(f"Warning: LLM returned page_number {resp_page_num} for detection, expected {page_number_display}.")
            # Decide if this is a critical error, for now we proceed if has_table is valid
        
        if has_table_val in ["YES", "NO"]: return has_table_val
        return "ERROR_VALUE"

    except json.JSONDecodeError: return "ERROR_JSON_DECODE"
    except ollama.ResponseError as e: print(f"Ollama API Error (Stage 1): {e}"); return "ERROR_OLLAMA_API"
    except Exception as e: print(f"Unexpected error (Stage 1): {e}"); return "ERROR_UNEXPECTED_CALL"

# --- Stage 2: Extract table data if table exists ---
def extract_table_data_from_page_ollama(model_name, image_base64, page_text, ollama_host_url, page_number_display):
    """
    Sends a page image and its extracted text to Ollama to extract structured table data.
    Expects a specific JSON format as output.
    """
    if not image_base64:
        return {"error": "ERROR_NO_IMAGE_FOR_EXTRACTION"}
    if not page_text:
        page_text = "No text extracted from this page." # Provide a fallback

    prompt = f"""
<Instructions>
You are an AI assistant specialized in extracting structured financial data from document pages.
You will receive a page image and the OCR'd text extracted from that same page.
The page number is {page_number_display}.
Your task is to analyze BOTH the image and the provided text to identify and extract data from the primary financial table on the page.

Carefully identify the following components:
1.  **{ResultKeys.PAGE_NUMBER.value}**: This MUST be {page_number_display}.
2.  **{ResultKeys.STATEMENT_TITLE.value}**: The main title of the financial statement (e.g., "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME", "CONSOLIDATED BALANCE SHEET"). If not present, use "N/A".
3.  **{ResultKeys.REPORTING_PERIOD.value}**: The reporting period line (e.g., "FOR THE YEAR ENDED 31 DECEMBER 2024", "AS AT 30 JUNE 2023"). If not present, use "N/A".
4.  **{ResultKeys.CURRENCY_SYMBOL.value}**: The currency symbol used (e.g., "$", "€", "AUD"). If multiple, pick the most prominent. If not clearly identifiable, use "N/A".
5.  **{ResultKeys.ROUNDING_SCALE.value}**: Any rounding information (e.g., "in thousands", "'000", "Millions", "Actual" if no rounding specified). If not present, assume "Actual".
6.  **{ResultKeys.COLUMN_HEADERS.value}**: An ordered list of strings representing the column headers of the main financial table. These headers will be used as keys in the 'values' dictionary for each line item.
7.  **{ResultKeys.LINE_ITEMS.value}**: A list of objects, where each object represents a row in the table. Each object must have:
    * **"{ResultKeys.DESCRIPTION.value}"**: The textual description of the financial line item (e.g., "Revenue", "Operating expenses").
    * **"{ResultKeys.NOTE_REFERENCE.value}"**: The note number or reference string associated with the line item (e.g., "3", "5(a)"). If no note, use an empty string "" or null.
    * **"{ResultKeys.VALUES.value}"**: A dictionary where keys are the exact strings from your identified `{ResultKeys.COLUMN_HEADERS.value}` list, and values are the corresponding string values from the table cells for that line item. All numerical values should be presented as strings. Example: {{"Note": "3", "2024 $": "38,403,987", "2023 $": "35,363,211"}}
    * **"{ResultKeys.IS_SUBTOTAL.value}"**: Boolean (true/false). True if the line item represents a subtotal.
    * **"{ResultKeys.IS_TOTAL.value}"**: Boolean (true/false). True if the line item represents a grand total or a major section total.
    * **"{ResultKeys.INDENTATION_LEVEL.value}"**: Integer (0, 1, 2, ...). Represents the indentation level of the description, indicating hierarchy. 0 for no indent.
</Instructions>

<InputTextFromPage>
{page_text}
</InputTextFromPage>

<OutputFormat>
Your response MUST be a single, valid JSON object adhering to the structure described above.
Ensure all string values within the JSON are properly escaped.
Example of the expected JSON structure:
{{
  "{ResultKeys.PAGE_NUMBER.value}": {page_number_display},
  "{ResultKeys.STATEMENT_TITLE.value}": "STATEMENT OF PROFIT OR LOSS AND OTHER COMPREHENSIVE INCOME",
  "{ResultKeys.REPORTING_PERIOD.value}": "FOR THE YEAR ENDED 31 DECEMBER 2024",
  "{ResultKeys.CURRENCY_SYMBOL.value}": "$",
  "{ResultKeys.ROUNDING_SCALE.value}": "Actual",
  "{ResultKeys.COLUMN_HEADERS.value}": ["Note", "2024 $", "2023 $"],
  "{ResultKeys.LINE_ITEMS.value}": [
    {{
      "{ResultKeys.DESCRIPTION.value}": "Revenue",
      "{ResultKeys.NOTE_REFERENCE.value}": "3",
      "{ResultKeys.VALUES.value}": {{"Note": "3", "2024 $": "38,403,987", "2023 $": "35,363,211"}},
      "{ResultKeys.IS_SUBTOTAL.value}": false,
      "{ResultKeys.IS_TOTAL.value}": false,
      "{ResultKeys.INDENTATION_LEVEL.value}": 0
    }},
    {{
      "{ResultKeys.DESCRIPTION.value}": "Cost of sales",
      "{ResultKeys.NOTE_REFERENCE.value}": "4",
      "{ResultKeys.VALUES.value}": {{"Note": "4", "2024 $": "(25,123,456)", "2023 $": "(22,987,654)"}},
      "{ResultKeys.IS_SUBTOTAL.value}": false,
      "{ResultKeys.IS_TOTAL.value}": false,
      "{ResultKeys.INDENTATION_LEVEL.value}": 0
    }},
    {{
      "{ResultKeys.DESCRIPTION.value}": "Gross profit",
      "{ResultKeys.NOTE_REFERENCE.value}": "",
      "{ResultKeys.VALUES.value}": {{"Note": "", "2024 $": "13,280,531", "2023 $": "12,375,557"}},
      "{ResultKeys.IS_SUBTOTAL.value}": true,
      "{ResultKeys.IS_TOTAL.value}": false,
      "{ResultKeys.INDENTATION_LEVEL.value}": 0
    }}
    // ... more line items
  ]
}}
Ensure your entire response is ONLY this JSON object.
</OutputFormat>
---
Process the image and text for page {page_number_display} and provide the structured JSON data.
"""
    print(f"\n[Stage 2] Sending Page {page_number_display} (Image + Text) to Ollama model '{model_name}' for data extraction...")

    try:
        client = ollama.Client(host=ollama_host_url)
        # Note: The 'text' part of the prompt is already included in the 'content'
        # The 'images' parameter handles the image data.
        response_data = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_base64]}],
            options={'temperature': 0.0}
        )
        print_llm_metrics(response_data, f"Page {page_number_display} (Data Extraction)")
        llm_output_str = strip_json_markdown(response_data['message']['content'])
        print(f"[Stage 2] Ollama processed response for Page {page_number_display}: '{llm_output_str[:500]}...' (truncated if long)")

        llm_json_response = json.loads(llm_output_str)
        
        # Basic validation
        if not isinstance(llm_json_response, dict):
            return {"error": "ERROR_EXTRACTION_FORMAT_NOT_DICT", "details": "LLM response was not a dictionary."}
        if int(str(llm_json_response.get(ResultKeys.PAGE_NUMBER.value))) != page_number_display:
            llm_json_response["warning"] = f"LLM returned page_number {llm_json_response.get(ResultKeys.PAGE_NUMBER.value)} for extraction, expected {page_number_display}."
        
        # TODO: Add more validation
        return llm_json_response

    except json.JSONDecodeError as e:
        return {"error": "ERROR_EXTRACTION_JSON_DECODE", "details": str(e), "raw_response": llm_output_str}
    except ollama.ResponseError as e:
        print(f"Ollama API Error (Stage 2): {e}")
        return {"error": "ERROR_EXTRACTION_OLLAMA_API", "details": str(e)}
    except Exception as e:
        print(f"Unexpected error (Stage 2): {e}")
        return {"error": "ERROR_EXTRACTION_UNEXPECTED_CALL", "details": str(e)}


# --- Main Orchestration ---
def detect_and_extract_tables_sequentially(pdf_path,
                                           model_name_param, ollama_host_param,
                                           image_dpi=150, total_pages_to_process_param=None):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return {}

    print(f"Starting sequential financial table detection and extraction for PDF: {pdf_path}")
    print(f"Ollama model: {model_name_param}")
    print(f"Image DPI: {image_dpi}")
    print("-" * 30)

    processed_page_results = {}
    doc = None

    try:
        doc = fitz.open(pdf_path)
        actual_total_pages_in_pdf = doc.page_count
        if actual_total_pages_in_pdf == 0: print("PDF is empty. Aborting."); return {}
        print(f"Total pages available in PDF: {actual_total_pages_in_pdf}")

        pages_to_process_count = actual_total_pages_in_pdf
        if total_pages_to_process_param is not None:
            pages_to_process_count = min(total_pages_to_process_param, actual_total_pages_in_pdf)
        print(f"Processing {pages_to_process_count} pages.")
        if pages_to_process_count == 0: print("No pages to process."); return {}

        for current_page_index in range(pages_to_process_count):
            page_num_internal = current_page_index
            page_num_display = current_page_index + 1

            print(f"\nProcessing Page {page_num_display} of {pages_to_process_count}...")
            img_base64 = convert_pdf_page_to_image_base64(doc, page_num_internal, dpi=image_dpi)

            if not img_base64:
                print(f"Failed to convert Page {page_num_display} to image. Marking as ERROR_CONVERSION.")
                processed_page_results[page_num_display] = "ERROR_CONVERSION"
                continue

            # Stage 1: Detect if page has a table
            table_detection_status = check_single_image_for_tables_ollama(
                model_name_param, img_base64, ollama_host_param, page_num_display
            )
            print(f"Page {page_num_display} - Table Detection Status: {table_detection_status}")

            if table_detection_status == "YES":
                # Stage 2: Extract table data
                print(f"Page {page_num_display} identified as containing a table. Proceeding to extraction...")
                page_obj = doc.load_page(page_num_internal)
                page_text = page_obj.get_text("text") # Extract text using PyMuPDF

                extracted_data = extract_table_data_from_page_ollama(
                    model_name_param, img_base64, page_text, ollama_host_param, page_num_display
                )
                processed_page_results[page_num_display] = extracted_data
                print(f"Page {page_num_display} - Extraction Result: {'Success (JSON returned)' if not extracted_data.get('error') else 'Failed (' + extracted_data.get('error', 'Unknown error') + ')'}")
            else:
                # Store the status from Stage 1 (e.g., "NO" or "ERROR_*")
                processed_page_results[page_num_display] = table_detection_status
            
            print("-" * 20)

    except Exception as e:
        print(f"Error opening or processing PDF with PyMuPDF: {e}")
    finally:
        if doc: doc.close()

    print("\n" + "=" * 30)
    print("Sequential Financial Table Detection and Extraction Complete.")
    print("=" * 30)
    return processed_page_results


if __name__ == "__main__":
    PDF_FILE_PATH = "docs/YHI PTY LTD.pdf"
    # For testing, process only a few pages, e.g., the first 5. Set to None to process all.
    TOTAL_PAGES_TO_PROCESS = 5 
    IMAGE_CONVERSION_DPI = 300


    results = detect_and_extract_tables_sequentially(
        pdf_path=PDF_FILE_PATH,
        model_name_param=OLLAMA_MULTIMODAL_MODEL,
        ollama_host_param=OLLAMA_HOST,
        image_dpi=IMAGE_CONVERSION_DPI,
        total_pages_to_process_param=TOTAL_PAGES_TO_PROCESS
    )

    print("\nFinal Sequential Financial Table Detection and Extraction Results:")
    for page_num in sorted(results.keys()):
        result_data = results[page_num]
        print(f"\n--- Page {page_num} ---")
        if isinstance(result_data, str): # Status from Stage 1 ("NO", "ERROR_*")
            print(f"  Status: {result_data}")
        elif isinstance(result_data, dict):
            if "error" in result_data:
                print(f"  Extraction Error: {result_data['error']}")
                if "details" in result_data: print(f"    Details: {result_data['details']}")
                if "raw_response" in result_data: print(f"    Raw LLM Output (if error): {result_data['raw_response'][:200]}...")
            else:
                print("  Extracted Data:")
                # Pretty print the JSON for readability
                print(json.dumps(result_data, indent=2))
        else:
            print(f"  Unexpected result type: {type(result_data)}")

    # Example: Save results to a JSON file
    output_json_path = os.path.join(pdf_dir if pdf_dir else ".", "extraction_results.json")
    try:
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {output_json_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")