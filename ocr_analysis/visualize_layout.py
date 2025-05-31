# This script provides functions to visualize the different stages of
# document layout analysis on an image. It helps in debugging and
# understanding the output of the layout analysis process.

from PIL import Image, ImageDraw, ImageFont
import os

def draw_boxes_on_image_direct(
    base_image_to_draw_on, # Should be a PIL Image object
    boxes_to_draw,         # List of boxes (either polygon or AA rect [xmin,ymin,xmax,ymax])
    color,
    thickness=2,
    labels=None,           # Can be True (auto-number), or a list of strings, or None
    font_path=None,
    label_font_size=15
):
    """
    Draws a list of boxes (polygons or rectangles) onto a PIL Image object.
    Optionally labels the boxes.

    Args:
        base_image_to_draw_on (PIL.Image.Image): The image to draw on.
        boxes_to_draw (list): List of boxes. Each box can be a polygon
                              (list of [x,y] points) or an AA rectangle
                              ([xmin, ymin, xmax, ymax]).
        color (str or tuple): Color for the box outlines and labels.
        thickness (int): Thickness of the box lines.
        labels (bool, list, optional): If True, labels boxes with their index.
                                       If a list, uses corresponding strings as labels.
                                       If None, no labels are drawn.
        font_path (str, optional): Path to a .ttf font file for labels.
        label_font_size (int): Font size for labels.

    Returns:
        PIL.Image.Image: The image with boxes drawn (modified in place).
    """
    draw = ImageDraw.Draw(base_image_to_draw_on) # Draw directly on the passed image
    label_font = None
    if labels: # If labels are requested (True or a list)
        resolved_font_path = font_path if font_path else "arial.ttf" # Default to Arial
        try:
            label_font = ImageFont.truetype(resolved_font_path, label_font_size)
        except IOError:
            # print(f"Warning: Font '{resolved_font_path}' not found. Using Pillow's default for labels.")
            try:
                label_font = ImageFont.load_default(size=label_font_size) # Pillow 10+
            except TypeError:
                label_font = ImageFont.load_default()


    for i, box_item in enumerate(boxes_to_draw):
        if not box_item: continue # Skip if box_item is None or empty

        text_label = ""
        if isinstance(labels, list) and i < len(labels):
            text_label = str(labels[i])
        elif labels is True: # Auto-number if labels is True
            text_label = str(i)

        try:
            # Check if it's a polygon (list of points)
            if isinstance(box_item[0], (list, tuple)) and len(box_item[0]) == 2:
                # Ensure coordinates are float for drawing
                poly_box_coords = [(float(p[0]), float(p[1])) for p in box_item]
                draw.polygon(poly_box_coords, outline=color, width=thickness)
                if text_label and label_font:
                    # Position label near the first point of the polygon, slightly above
                    draw.text((poly_box_coords[0][0], poly_box_coords[0][1] - (label_font_size + 5)),
                              text_label, fill=color, font=label_font)
            # Check if it's an axis-aligned rectangle [xmin, ymin, xmax, ymax]
            elif len(box_item) == 4:
                xmin, ymin, xmax, ymax = [float(c) for c in box_item]
                draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=thickness)
                if text_label and label_font:
                    # Position label near the top-left corner, slightly above
                    draw.text((xmin, ymin - (label_font_size + 5)),
                              text_label, fill=color, font=label_font)
            # else:
                # print(f"Warning: Skipping unrecognized box format: {box_item}")
        except (TypeError, ValueError, IndexError) as e:
            # print(f"Warning: Error drawing box {box_item}: {e}")
            pass # Continue to the next box
    return base_image_to_draw_on


def visualize_layout_stages_on_image(
    image_path,             # Path to the original image
    initial_ocr_results,    # List of (polygon_box, (text, score)) from OCR
    full_length_single_col_rows, # AA boxes [xmin,ymin,xmax,ymax]
    spanning_row_boxes,        # AA boxes
    identified_cols,           # List of (col_xmin, col_xmax)
    output_dir="layout_visualization_on_image",
    base_font_path=None     # e.g., "arial.ttf" or path to specific .ttf
):
    """
    Generates and saves a series of images visualizing the different stages
    of document layout analysis.

    Args:
        image_path (str): Path to the original source image.
        initial_ocr_results (list): OCR results as [(poly_box, (text, score)), ...].
                                    Used to draw initial text boxes.
        full_length_single_col_rows (list): AA boxes for rows within columns.
        spanning_row_boxes (list): AA boxes for rows that span columns.
        identified_cols (list): Tuples (col_xmin, col_xmax) for identified columns.
        output_dir (str): Directory to save the visualization images.
        base_font_path (str, optional): Path to a .ttf font file for titles/labels.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found for visualization: {image_path}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    source_image = Image.open(image_path).convert("RGBA") # Use RGBA for transparency handling
    img_w, img_h = source_image.size

    # Determine font sizes relative to image height, with min sizes
    title_font_size = max(15, int(img_h * 0.03))
    label_font_size = max(12, int(title_font_size * 0.8))
    
    resolved_font_path = base_font_path if base_font_path else "arial.ttf"
    try:
        title_font = ImageFont.truetype(resolved_font_path, title_font_size)
    except IOError:
        # print(f"Title font '{resolved_font_path}' not found, using default.")
        try:
            title_font = ImageFont.load_default(size=title_font_size)
        except TypeError:
            title_font = ImageFont.load_default()

    initial_polygon_boxes = [item[0] for item in initial_ocr_results]

    # --- Stage 1: Initial OCR Polygons ---
    img_s1_rgba = source_image.copy() # Work on a copy
    draw_boxes_on_image_direct(img_s1_rgba, initial_polygon_boxes, color='blue', labels=True, 
                               font_path=base_font_path, label_font_size=label_font_size)
    ImageDraw.Draw(img_s1_rgba).text((10,10), "Stage 1: Initial OCR Polygons", fill="black", font=title_font)
    # Composite onto a white background if saving as JPG/PNG without transparency issues
    img_s1_rgb = Image.new("RGB", img_s1_rgba.size, "white")
    img_s1_rgb.paste(img_s1_rgba, mask=img_s1_rgba.split()[3] if img_s1_rgba.mode == "RGBA" else None)
    s1_path = os.path.join(output_dir, "01_initial_polygons.png")
    img_s1_rgb.save(s1_path); print(f"Saved: {s1_path}")

    # --- Stage 2: Identified Columns ---
    img_s2_rgba = source_image.copy()
    draw_s2_text_overlay = ImageDraw.Draw(img_s2_rgba) # For drawing text labels on top of transparent fills
    if identified_cols:
        # Define a list of semi-transparent fill colors for columns
        col_fills_rgba = [
            (255, 0, 0, 40), (0, 255, 0, 40), (0, 0, 255, 40), 
            (255, 255, 0, 40), (255, 0, 255, 40), (0, 255, 255, 40)
        ]
        for i, (cx1, cx2) in enumerate(identified_cols):
            fill_color_rgba = col_fills_rgba[i % len(col_fills_rgba)]
            outline_color_rgb = tuple(c // 2 for c in fill_color_rgba[:3]) # Darker outline

            # Create a transparent layer for this column's fill
            col_fill_layer = Image.new('RGBA', img_s2_rgba.size, (0,0,0,0)) # Fully transparent
            draw_col_fill = ImageDraw.Draw(col_fill_layer)
            draw_col_fill.rectangle([cx1, 0, cx2, img_h], fill=fill_color_rgba, outline=outline_color_rgb, width=1)
            
            # Alpha composite this layer onto the current stage image
            img_s2_rgba = Image.alpha_composite(img_s2_rgba, col_fill_layer)
            
            # Draw column label text on the main image (after compositing fills)
            if title_font:
                draw_s2_text_overlay.text((cx1 + 5, img_h * 0.05), f"Col {i+1}", fill="black", font=title_font)
    
    ImageDraw.Draw(img_s2_rgba).text((10,10), "Stage 2: Identified Columns", fill="black", font=title_font)
    img_s2_rgb = Image.new("RGB", img_s2_rgba.size, "white")
    img_s2_rgb.paste(img_s2_rgba, mask=img_s2_rgba.split()[3] if img_s2_rgba.mode == "RGBA" else None)
    s2_path = os.path.join(output_dir, "02_identified_columns.png")
    img_s2_rgb.save(s2_path); print(f"Saved: {s2_path}")
    
    # --- Stage 3: Spanning Rows ---
    img_s3_rgba = source_image.copy()
    if identified_cols: # Draw faint column lines for context
        draw_faint_cols = ImageDraw.Draw(img_s3_rgba)
        for cx1,cx2 in identified_cols:
            draw_faint_cols.line([(cx1,0),(cx1,img_h)],fill=(200,200,200,100),width=1) # Faint gray lines
            draw_faint_cols.line([(cx2,0),(cx2,img_h)],fill=(200,200,200,100),width=1)
    draw_boxes_on_image_direct(img_s3_rgba, spanning_row_boxes, color='purple', thickness=3, labels=True,
                               font_path=base_font_path, label_font_size=label_font_size)
    ImageDraw.Draw(img_s3_rgba).text((10,10), "Stage 3: Spanning Rows", fill="black", font=title_font)
    img_s3_rgb = Image.new("RGB", img_s3_rgba.size, "white")
    img_s3_rgb.paste(img_s3_rgba, mask=img_s3_rgba.split()[3] if img_s3_rgba.mode == "RGBA" else None)
    s3_path = os.path.join(output_dir, "03_spanning_rows.png")
    img_s3_rgb.save(s3_path); print(f"Saved: {s3_path}")

    # --- Stage 4: Full Column Width Rows (In-Column Rows) ---
    img_s4_rgba = source_image.copy()
    if identified_cols: # Draw faint column lines
        draw_faint_cols_s4 = ImageDraw.Draw(img_s4_rgba)
        for cx1,cx2 in identified_cols:
            draw_faint_cols_s4.line([(cx1,0),(cx1,img_h)],fill=(200,200,200,100),width=1)
            draw_faint_cols_s4.line([(cx2,0),(cx2,img_h)],fill=(200,200,200,100),width=1)
    draw_boxes_on_image_direct(img_s4_rgba, full_length_single_col_rows, color='red', thickness=2, labels=True,
                               font_path=base_font_path, label_font_size=label_font_size)
    ImageDraw.Draw(img_s4_rgba).text((10,10), "Stage 4: Full-Column-Width Rows", fill="black", font=title_font)
    img_s4_rgb = Image.new("RGB", img_s4_rgba.size, "white")
    img_s4_rgb.paste(img_s4_rgba, mask=img_s4_rgba.split()[3] if img_s4_rgba.mode == "RGBA" else None)
    s4_path = os.path.join(output_dir, "04_full_column_rows.png")
    img_s4_rgb.save(s4_path); print(f"Saved: {s4_path}")

    # --- Stage 5: Combined Final Layout ---
    img_s5_rgba = source_image.copy()
    draw_s5_text_overlay = ImageDraw.Draw(img_s5_rgba) # For column labels
    if identified_cols: # Draw column fills first
        for i,(cx1,cx2) in enumerate(identified_cols):
            fill_color_rgba = col_fills_rgba[i % len(col_fills_rgba)]
            col_fill_layer_s5 = Image.new('RGBA', img_s5_rgba.size, (0,0,0,0))
            ImageDraw.Draw(col_fill_layer_s5).rectangle([cx1,0,cx2,img_h], fill=fill_color_rgba, outline=None) # No outline for fill
            img_s5_rgba = Image.alpha_composite(img_s5_rgba, col_fill_layer_s5)
            if title_font: # Draw column labels on top of fills
                draw_s5_text_overlay.text((cx1+5, img_h * 0.05),f"Col {i+1}", fill="black", font=title_font)
    
    # Then draw spanning rows
    draw_boxes_on_image_direct(img_s5_rgba, spanning_row_boxes, color='purple', thickness=3, labels=True,
                               font_path=base_font_path, label_font_size=label_font_size)
    # Then draw in-column rows (full width)
    draw_boxes_on_image_direct(img_s5_rgba, full_length_single_col_rows, color='darkred', thickness=2, labels=True,
                               font_path=base_font_path, label_font_size=label_font_size)
    
    ImageDraw.Draw(img_s5_rgba).text((10,10), "Stage 5: Combined Layout", fill="black", font=title_font)
    img_s5_rgb = Image.new("RGB", img_s5_rgba.size, "white")
    img_s5_rgb.paste(img_s5_rgba, mask=img_s5_rgba.split()[3] if img_s5_rgba.mode == "RGBA" else None)
    s5_path = os.path.join(output_dir, "05_combined_layout.png")
    img_s5_rgb.save(s5_path); print(f"Saved: {s5_path}")