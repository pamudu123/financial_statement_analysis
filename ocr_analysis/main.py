# This script demonstrates the full pipeline:
# 1. Perform OCR on an image using functions from ocr_utils.py.
# 2. Analyze the document layout (columns, rows) using functions from layout_analysis.py.
# 3. Visualize the different stages of layout analysis using functions from visualize_layout.py.

import os
from PIL import Image, ImageDraw, ImageFont # For dummy image creation if needed

# Import functions from the other created modules
from ocr_utils import predict_and_visualize_ocr
from layout_analysis import process_document_layout_with_ocr
from visualize_layout import visualize_layout_stages_on_image

def main():
    print("Starting Document Layout Analysis Pipeline...")

    # --- Configuration ---
    image_file_path = "docs/page11.jpg"
    ocr_language = 'en'
    font_for_drawing = None 

    # Directory to save the output visualization images
    output_visualization_dir = "output/document_layouts"
    if not os.path.exists(output_visualization_dir):
        os.makedirs(output_visualization_dir)
        print(f"Created output directory: {output_visualization_dir}")

    # --- Step 1: Perform OCR and Initial Visualization ---
    print(f"\n--- Step 1: Performing OCR on '{image_file_path}' ---")
    
    ocr_results_data = predict_and_visualize_ocr(
        image_path=image_file_path,
        lang=ocr_language,
        font_path=font_for_drawing,
        use_custom_draw=True # Use the more detailed custom drawing for initial OCR viz
    )

    if not ocr_results_data or not ocr_results_data["ocr_data"]:
        print("OCR process failed or no text detected. Cannot proceed with layout analysis.")
        return

    # Save the initially annotated OCR image
    if ocr_results_data["image_annotated"]:
        ocr_output_path = os.path.join(output_visualization_dir, f"00_ocr_annotated_{os.path.basename(image_file_path)}")
        ocr_results_data["image_annotated"].save(ocr_output_path)
        print(f"Initial OCR annotated image saved to: {ocr_output_path}")

    # Prepare OCR results for layout analysis: list of (polygon_box, (text_content, score))
    # The `process_document_layout_with_ocr` function expects this format.
    # `ocr_results_data["ocr_data"]` is like: [[[points, (text, confidence)], ...]] for a single image.
    
    ocr_data_for_layout = []
    if ocr_results_data["ocr_data"] and ocr_results_data["ocr_data"][0]:
        lines = ocr_results_data["ocr_data"][0] # For a single image result
        for line_info in lines:
            # line_info is [points, (text, confidence)]
            polygon_box = line_info[0]
            text_content = line_info[1][0]
            score = line_info[1][1]
            ocr_data_for_layout.append((polygon_box, (text_content, score)))
    
    if not ocr_data_for_layout:
        print("No valid OCR data extracted to proceed with layout analysis.")
        return

    # --- Step 2: Process Document Layout ---
    print(f"\n--- Step 2: Processing Document Layout ---")
    
    # Parameters for layout analysis (can be tuned)
    # Refer to layout_analysis.py for details on these parameters.
    layout_params = {
        "word_expansion_factor_global": 0.05, # Smaller global expansion can be better
        "row_v_overlap_ratio_global": 0.3,
        "word_expansion_factor_in_col": 0.01, # Minimal expansion within columns
        "row_v_overlap_ratio_in_col": 0.3,
        "col_id_method": 'keywords_else_simple',  # Try keyword-based first, then fallback
        "col_note_keyword": "Note",            # Keyword for the 'Note' column
        # Fallback parameters for 'simple_only' or if keyword method fails:
        "col_smooth_window": 5,
        "col_gap_thresh_factor": 0.02,        # More sensitive to small gaps for fallback
        "col_min_width_pixels": 20            # Smaller min width for columns in fallback
    }

    full_col_rows, spanning_rows, identified_cols = process_document_layout_with_ocr(
        ocr_results_tuples=ocr_data_for_layout,
        **layout_params
    )

    print(f"Layout analysis complete.")
    print(f" - Identified Columns: {len(identified_cols) if identified_cols else 0}")
    # print(f"   Details: {identified_cols}")
    print(f" - Spanning Rows: {len(spanning_rows)}")
    print(f" - Full Column-Width Rows: {len(full_col_rows)}")


    # --- Step 3: Visualize Layout Stages ---
    print(f"\n--- Step 3: Visualizing Layout Stages ---")
    
    # The initial_ocr_results for visualization needs to be in the format:
    # [(poly_box, (text, score)), ...] which `ocr_data_for_layout` already is.
    visualize_layout_stages_on_image(
        image_path=image_file_path,
        initial_ocr_results=ocr_data_for_layout, 
        full_length_single_col_rows=full_col_rows,
        spanning_row_boxes=spanning_rows,
        identified_cols=identified_cols,
        output_dir=output_visualization_dir,
        base_font_path=font_for_drawing 
    )
    
    print(f"All visualization images saved in '{os.path.abspath(output_visualization_dir)}'")
    print("Document Layout Analysis Pipeline Finished.")

if __name__ == "__main__":
    main()