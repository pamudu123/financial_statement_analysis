import fitz  # PyMuPDF
import base64

# --- Configuration ---
OLLAMA_MULTIMODAL_MODEL = 'gemma3:4b'  # Or your preferred Ollama multimodal model
OLLAMA_HOST = 'http://localhost:11434'

# --- Helper Functions ---
def convert_pdf_page_to_image_base64(pdf_doc, page_number_internal, dpi=150):
    """
    Converts a single PDF page to a base64 encoded image string using PyMuPDF.
    """
    try:
        page = pdf_doc.load_page(page_number_internal)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_bytes = pix.tobytes("png")
        if img_bytes:
            return base64.b64encode(img_bytes).decode("utf-8")
        else:
            print(f"Error: Could not convert page {page_number_internal + 1} to image bytes.")
            return None
    except Exception as e:
        print(f"Error converting PDF page {page_number_internal + 1} to image: {e}")
        return None

def convert_pdf_page_to_image_and_text(pdf_doc, page_number_internal, dpi=150):
    """
    Converts a single PDF page to a base64 encoded image string and extracts its text using PyMuPDF.
    Returns a tuple (base64_image_string, extracted_text) or (None, extracted_text_if_any) if image conversion fails.
    Text extraction will be attempted even if image conversion fails.
    """
    page_obj = None
    base64_image = None
    extracted_text = ""

    try:
        page_obj = pdf_doc.load_page(page_number_internal)

        # Image conversion
        try:
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page_obj.get_pixmap(matrix=matrix, alpha=False)
            img_bytes = pix.tobytes("png")
            if img_bytes:
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
            else:
                print(f"Warning: Could not convert page {page_number_internal + 1} to image bytes.")
        except Exception as img_e:
            print(f"Error during image conversion for page {page_number_internal + 1}: {img_e}")

        # Text extraction
        try:
            extracted_text = page_obj.get_text("text")
            if not extracted_text.strip(): # If basic text extraction is empty, try blocks
                blocks = page_obj.get_text("blocks")
                text_from_blocks = []
                for block in blocks:
                    if len(block) > 4 and isinstance(block[4], str):
                        text_from_blocks.append(block[4].strip())
                extracted_text = "\n".join(filter(None, text_from_blocks))
            if not extracted_text.strip():
                print(f"Warning: Text extraction resulted in empty text for page {page_number_internal + 1}.")
        except Exception as text_e:
            print(f"Error during text extraction for page {page_number_internal + 1}: {text_e}")
            extracted_text = "" # Ensure it's an empty string on error

        # If image conversion failed critically, base64_image will be None
        if base64_image is None:
             print(f"Info: Image conversion failed for page {page_number_internal + 1}, but text extraction might have succeeded.")
        
        return base64_image, extracted_text

    except Exception as e:
        print(f"Overall error processing PDF page {page_number_internal + 1} for image/text: {e}")
        # Attempt to return any partial success if page_obj was loaded
        current_text = extracted_text # Use already attempted extracted_text
        if page_obj and not current_text:
            try:
                current_text = page_obj.get_text("text")
                if not current_text.strip():
                    blocks = page_obj.get_text("blocks")
                    text_from_blocks = []
                    for block in blocks:
                        if len(block) > 4 and isinstance(block[4], str):
                            text_from_blocks.append(block[4].strip())
                    current_text = "\n".join(filter(None, text_from_blocks))
            except:
                pass # Ignore error during fallback text extraction
        return base64_image, current_text # base64_image could be None


def print_llm_metrics(response_data, context_message="LLM Call"):
    """Prints standardized LLM call metrics."""
    prompt_tokens = response_data.get('prompt_eval_count', 'N/A')
    completion_tokens = response_data.get('eval_count', 'N/A')
    total_duration_ms = response_data.get('total_duration', 0) / 1_000_000 # Convert nanoseconds to ms
    load_duration_ms = response_data.get('load_duration', 0) / 1_000_000
    prompt_eval_duration_ms = response_data.get('prompt_eval_duration', 0) / 1_000_000
    eval_duration_ms = response_data.get('eval_duration', 0) / 1_000_000

    print(f"LLM Call Metrics ({context_message}):")
    print(f"  - Load Duration: {load_duration_ms:.2f} ms")
    print(f"  - Prompt Tokens: {prompt_tokens} (Duration: {prompt_eval_duration_ms:.2f} ms)")
    print(f"  - Completion Tokens: {completion_tokens} (Duration: {eval_duration_ms:.2f} ms)")
    print(f"  - Total Duration: {total_duration_ms:.2f} ms")

def strip_json_markdown(llm_output_str):
    """Strips ```json ... ``` or ``` ... ``` markdown from LLM string output."""
    if llm_output_str.startswith("```json"):
        return llm_output_str.lstrip("```json").rstrip("```").strip()
    elif llm_output_str.startswith("```"):
        return llm_output_str.lstrip("```").rstrip("```").strip()
    return llm_output_str