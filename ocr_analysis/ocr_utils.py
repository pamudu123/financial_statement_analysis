# This script contains utility functions for performing OCR using PaddleOCR
# and visualizing the raw OCR results on an image.

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import os

def custom_draw_ocr(image, boxes, txts, scores,
                    font_path=None,
                    default_font_name="arial.ttf", # A common font, change if needed
                    font_size_ratio=0.8, # Ratio of box height for font size
                    min_font_size=12,    # Minimum font size
                    box_color='red',
                    text_color='white',
                    text_bg_color='red',
                    box_thickness=2,
                    text_margin=3): # Margin between box and text background
    """
    Draws OCR results (bounding boxes, text, scores) on a PIL Image with custom styling.

    Args:
        image (PIL.Image.Image): The input image.
        boxes (list): A list of bounding boxes. Each box is a list of 4 points,
                      e.g., [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
        txts (list): A list of recognized text strings.
        scores (list): A list of confidence scores for the recognized texts.
        font_path (str, optional): Path to a .ttf font file. If None, tries
                                   default_font_name then Pillow's default.
        default_font_name (str, optional): Name of a default system font to try.
        font_size_ratio (float, optional): Factor to scale font size by box height.
        min_font_size (int, optional): Minimum font size.
        box_color (str, optional): Color for the bounding boxes.
        text_color (str, optional): Color for the recognized text.
        text_bg_color (str, optional): Background color for the text.
        box_thickness (int, optional): Thickness of the bounding box lines.
        text_margin (int, optional): Margin between the top of the box and text background.

    Returns:
        PIL.Image.Image: A new image with OCR results drawn.
    """
    if not boxes: # If there are no boxes, return the original image
        return image.copy()

    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for i in range(len(boxes)):
        current_box = boxes[i] # List of 4 [x,y] points
        current_text = txts[i]
        current_score = scores[i]

        # Ensure current_box points are tuples for Pillow
        # And that they are valid numbers
        try:
            pil_box = [tuple(map(float, p)) for p in current_box]
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping invalid box data: {current_box}. Error: {e}")
            continue

        # Draw the bounding box polygon
        draw.polygon(pil_box, outline=box_color, width=box_thickness)

        # Calculate an approximate height of the box for font scaling
        try:
            box_ymin = min(p[1] for p in pil_box)
            box_ymax = max(p[1] for p in pil_box)
            box_height = box_ymax - box_ymin
        except ValueError: # Handles empty pil_box if points were invalid
            box_height = min_font_size / font_size_ratio # Fallback height

        font_size = max(min_font_size, int(box_height * font_size_ratio))

        # Load font
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Warning: Could not load font at {font_path}. Trying default.")
        
        if not font: # If font_path failed or was not provided
            try:
                font = ImageFont.truetype(default_font_name, font_size)
            except IOError:
                # print(f"Warning: Could not load default font {default_font_name}. Using Pillow's default.")
                try:
                    font = ImageFont.load_default(size=font_size) # Pillow 10+ allows size
                except TypeError: # older Pillow
                    font = ImageFont.load_default()


        display_text = f"{current_text} ({current_score:.2f})"

        # Get text bounding box relative to (0,0) to determine its width and height
        try:
            # font.getbbox is preferred for modern Pillow
            text_bbox_at_origin = font.getbbox(display_text) # (left, top, right, bottom)
            text_width = text_bbox_at_origin[2] - text_bbox_at_origin[0]
            text_height = text_bbox_at_origin[3] - text_bbox_at_origin[1]
            # text_offset_y_from_origin_top is the distance from the drawing y-coordinate to the top of the ink
            text_offset_y_from_origin_top = text_bbox_at_origin[1]
        except AttributeError: 
            # Fallback for older Pillow or if font is load_default() in some cases
            # draw.textsize is deprecated but a common fallback
            try:
                text_size_dep = draw.textsize(display_text, font=font)
            except AttributeError: # if font is from load_default() it might not have getbbox
                # A very rough estimate if all else fails
                text_size_dep = (len(display_text) * (font_size // 2 if hasattr(font, 'size') and font.size else min_font_size // 2), 
                                 font_size if hasattr(font, 'size') and font.size else min_font_size)
            text_width = text_size_dep[0]
            text_height = text_size_dep[1]
            text_offset_y_from_origin_top = 0 # Assume text starts at y=0 for this fallback


        # Define the top-left corner for the text background
        # Position text background above the top-left point of the bounding box
        # pil_box[0] is often the top-left point for horizontal text in many OCR outputs
        text_anchor_x = pil_box[0][0]
        text_anchor_y = pil_box[0][1]

        bg_x0 = text_anchor_x
        bg_y1 = text_anchor_y - text_margin      # Bottom of the background rect
        bg_y0 = bg_y1 - text_height              # Top of the background rect
        bg_x1 = text_anchor_x + text_width

        # Draw background rectangle
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=text_bg_color)

        # Define the text drawing position (top-left of where text rendering starts)
        # Adjust for the font's internal bbox 'top' offset
        draw_text_x = text_anchor_x
        draw_text_y = bg_y0 - text_offset_y_from_origin_top

        draw.text((draw_text_x, draw_text_y), display_text, font=font, fill=text_color)

    return img_copy

def predict_and_visualize_ocr(image_path, lang='en', font_path=None, use_custom_draw=True):
    """
    Performs OCR on an image using PaddleOCR, visualizes the bounding boxes, 
    and returns the annotated image along with OCR data.

    Args:
        image_path (str): Path to the input image.
        lang (str): Language code for OCR (e.g., 'en', 'ch', 'fr', 'german', 'korean', 'japan').
                    Refer to PaddleOCR documentation for all supported languages.
        font_path (str, optional): Path to a .ttf font file for drawing text.
                                   If None, PaddleOCR's default or `custom_draw_ocr`'s default is used.
                                   Example: "arial.ttf" or "./simfang.ttf".
        use_custom_draw (bool, optional): If True, uses `custom_draw_ocr` for visualization.
                                          If False, uses PaddleOCR's built-in `draw_ocr`.

    Returns:
        dict: A dictionary containing:
            - "image_annotated" (PIL.Image.Image): Image with OCR results drawn, or None if an error.
            - "ocr_data" (list): Raw result from ocr_engine.ocr(), or None.
            - "boxes" (list): List of bounding boxes, or None.
            - "texts" (list): List of recognized texts, or None.
            - "scores" (list): List of confidence scores, or None.
        Returns None for all values in dict if a major error occurs or no text is found.
    """
    print(f"Initializing PaddleOCR for language: {lang}...")
    try:
        # Initialize PaddleOCR.
        # - use_angle_cls=True: enable text angle classification.
        # - lang: specify the language.
        # - use_gpu=True/False: set to True if you installed paddlepaddle-gpu and have a GPU.
        # - show_log=False: suppress verbose PaddleOCR logging.
        ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False, show_log=False)
        print(f"PaddleOCR initialized. Processing image: {image_path}")

        if not os.path.exists(image_path):
            print(f"Error: Image file not found at '{image_path}'")
            return {
                "image_annotated": None, "ocr_data": None, "boxes": None, 
                "texts": None, "scores": None
            }

        # Perform OCR
        ocr_raw_result = ocr_engine.ocr(image_path, cls=True)

        if not ocr_raw_result or not ocr_raw_result[0]: # Check if result is None or empty
            print(f"No text detected in '{image_path}' for language '{lang}'.")
            return {
                "image_annotated": None, "ocr_data": ocr_raw_result, "boxes": [], 
                "texts": [], "scores": [] # Return empty lists for consistency
            }
        
        print(f"Text detected. Visualizing results...")

        # Load the original image using Pillow
        image = Image.open(image_path).convert('RGB')

        # Extract data for visualization
        # ocr_raw_result is a list of lists, e.g. [[[points, (text, confidence)], ...]]
        # For a single image, we take the first element ocr_raw_result[0]
        lines = ocr_raw_result[0]
        boxes = [line[0] for line in lines]
        txts = [line[1][0] for line in lines]
        scores = [line[1][1] for line in lines]

        annotated_image = None
        if use_custom_draw:
            # Use the enhanced custom_draw_ocr function
            annotated_image = custom_draw_ocr(image, boxes, txts, scores, font_path=font_path)
        else:
            # Use PaddleOCR's built-in draw_ocr (less customizable text rendering)
            # Note: PaddleOCR's draw_ocr might require the font path to be accessible by its internal logic.
            # It typically defaults to 'simfang.ttf' if font_path is None or not found.
            from paddleocr import draw_ocr # Import locally if only used here
            annotated_image = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        
        results_dict = {
            "image_annotated": annotated_image,
            "ocr_data": ocr_raw_result, # Full raw result
            "boxes": boxes,
            "texts": txts,
            "scores": scores
        }
        print("OCR and visualization complete.")
        return results_dict

    except Exception as e:
        print(f"An error occurred during OCR processing or visualization: {e}")
        import traceback
        traceback.print_exc()
        return {
            "image_annotated": None, "ocr_data": None, "boxes": None, 
            "texts": None, "scores": None
        }

if __name__ == '__main__':
    print("Testing ocr_utils.py...")
    
    # --- Configuration for testing ---
    # 1. Specify the path to your image
    #    Replace with your image path.
    test_image_path = "docs/page11.jpg"

    # 2. Specify the language for OCR
    language_code = 'en' # English

    # 3. Specify the font path for visualization (optional)
    #    For custom_draw_ocr or PaddleOCR's draw_ocr.
    #    - Set to None to use defaults.
    #    - Or provide a path to a .ttf font file, e.g., "arial.ttf"
    #    - You might need to download 'simfang.ttf' for good CJK character rendering
    #      from PaddleOCR's GitHub and place it in your script's directory or provide the full path.
    # custom_font_for_drawing = "simfang.ttf" # Example for CJK
    custom_font_for_drawing = None # Will try "arial.ttf" then Pillow's default

    # --- Perform OCR and Visualization ---
    if os.path.exists(test_image_path):
        ocr_results = predict_and_visualize_ocr(
            image_path=test_image_path,
            lang=language_code,
            font_path=custom_font_for_drawing,
            use_custom_draw=True # Set to False to test PaddleOCR's draw_ocr
        )
        
        if ocr_results and ocr_results["image_annotated"]:
            annotated_image_pil = ocr_results["image_annotated"]
            
            # To display the image
            annotated_image_pil.show()

            # To save the annotated image
            output_filename = f"output_ocr_annotated_{os.path.basename(test_image_path)}"
            annotated_image_pil.save(output_filename)
            print(f"✅ Annotated OCR image saved to '{output_filename}'")
            
            # print("\nOCR Detected Texts:")
            # if ocr_results["texts"]:
            #     for i, txt in enumerate(ocr_results["texts"]):
            #         print(f"- \"{txt}\" (Score: {ocr_results['scores'][i]:.2f})")
            # else:
            #     print("No texts were extracted or available in results.")
        else:
            print(f"❌ Could not process the image for OCR or no text was found.")
    else:
        print(f"Test image file not found: '{test_image_path}'. Please check the path or ensure dummy image creation was successful.")

    print("\nocr_utils.py test finished.")