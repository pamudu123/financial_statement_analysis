# This script provides functions for analyzing the layout of a document
# based on OCR results. It includes methods for identifying columns (keyword-based
# and simple projection), merging text boxes into rows, and distinguishing
# spanning elements from single-column elements.

import math
import numpy as np

# --- Core Helper Functions for Bounding Box Manipulation ---

def convert_to_axis_aligned(poly_box):
    """
    Converts a 4-point polygon box to an axis-aligned [xmin, ymin, xmax, ymax] box.
    A polygon box is a list of 4 points, e.g., [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
    """
    if not poly_box or not isinstance(poly_box, list) or len(poly_box) != 4:
        # print(f"Warning: Invalid polygon box format: {poly_box}")
        return None
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in poly_box):
        # print(f"Warning: Invalid point format in polygon box: {poly_box}")
        return None
    try:
        x_coords = [float(p[0]) for p in poly_box]
        y_coords = [float(p[1]) for p in poly_box]
    except (ValueError, TypeError):
        # print(f"Warning: Invalid coordinate value in polygon box: {poly_box}")
        return None
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def expand_box_horizontally(box, expansion_factor_each_side):
    """
    Expands an axis-aligned box [xmin, ymin, xmax, ymax] horizontally on both sides.
    Expansion is proportional to the box's width.
    """
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    if width < 0:  # Should not happen with correctly ordered xmin, xmax
        # print(f"Warning: Box has negative width: {box}")
        return box # Return original if width is invalid
    extension = width * expansion_factor_each_side
    return [xmin - extension, ymin, xmax + extension, ymax]

def check_vertical_alignment(box1_aa, box2_aa, vertical_overlap_threshold_ratio):
    """
    Checks if two axis-aligned boxes are sufficiently vertically aligned.
    Alignment is based on the ratio of their vertical overlap to the height of the smaller box.

    Args:
        box1_aa, box2_aa: Axis-aligned boxes [xmin, ymin, xmax, ymax].
        vertical_overlap_threshold_ratio (float): Minimum overlap ratio to be considered aligned.
                                                 E.g., 0.3 means at least 30% overlap.
    Returns:
        bool: True if vertically aligned, False otherwise.
    """
    y1_b1, y2_b1 = box1_aa[1], box1_aa[3]  # ymin, ymax for box1
    y1_b2, y2_b2 = box2_aa[1], box2_aa[3]  # ymin, ymax for box2

    height_b1 = y2_b1 - y1_b1
    height_b2 = y2_b2 - y1_b2

    epsilon = 1e-6  # A small number to handle floating point inaccuracies and zero height
    if height_b1 <= epsilon or height_b2 <= epsilon:
        return False # Cannot determine alignment for zero-height boxes

    # Calculate vertical overlap
    overlap_y_start = max(y1_b1, y1_b2)
    overlap_y_end = min(y2_b1, y2_b2)
    overlap_height = overlap_y_end - overlap_y_start

    if overlap_height <= epsilon:  # No significant vertical overlap
        return False

    min_box_height = min(height_b1, height_b2)
    
    if min_box_height <= epsilon: # Should be caught by individual height checks, but good for safety
        return False
        
    overlap_ratio = overlap_height / min_box_height
    return overlap_ratio > vertical_overlap_threshold_ratio

def create_rows_from_boxes(
    initial_polygon_boxes,
    horizontal_expansion_factor_each_side=0.15,
    vertical_overlap_threshold_ratio=0.3
):
    """
    Converts polygon boxes to axis-aligned (AA) format, expands them horizontally,
    sorts them, and merges them into rows based on vertical alignment and horizontal overlap.

    Args:
        initial_polygon_boxes (list): List of polygon boxes from OCR.
        horizontal_expansion_factor_each_side (float): Factor to expand AA boxes horizontally.
        vertical_overlap_threshold_ratio (float): Threshold for vertical alignment check.

    Returns:
        list: A list of merged AA boxes [xmin, ymin, xmax, ymax] representing the formed rows.
    """
    if not initial_polygon_boxes:
        return []
    
    # Convert to AA and filter out invalid or zero-area boxes
    axis_aligned_boxes = []
    for poly_box in initial_polygon_boxes:
        aa_box = convert_to_axis_aligned(poly_box)
        # Ensure the box has positive width and height (greater than a small epsilon)
        if aa_box and (aa_box[2] - aa_box[0] > 1e-6) and \
           (aa_box[3] - aa_box[1] > 1e-6):
            axis_aligned_boxes.append(aa_box)
            
    if not axis_aligned_boxes:
        return []

    # Expand boxes horizontally to encourage merging of nearby words into a line
    extended_boxes = [expand_box_horizontally(b, horizontal_expansion_factor_each_side) for b in axis_aligned_boxes]
    
    # Sort boxes primarily by y_min (top coordinate), then by x_min (left coordinate)
    # This helps process boxes in a top-to-bottom, left-to-right reading order.
    sorted_boxes = sorted(extended_boxes, key=lambda b: (b[1], b[0]))
    
    final_rows_aa = []
    if not sorted_boxes: # Should not happen if axis_aligned_boxes was populated
        return []

    current_row_box = list(sorted_boxes[0]) # Start with the first box as the beginning of a row

    for i in range(1, len(sorted_boxes)):
        next_box = sorted_boxes[i]
        
        # Check for horizontal overlap between the current row's extent and the next box
        horizontally_overlaps = max(current_row_box[0], next_box[0]) < min(current_row_box[2], next_box[2])
        # Check for vertical alignment
        vertically_aligned = check_vertical_alignment(current_row_box, next_box, vertical_overlap_threshold_ratio)

        if vertically_aligned and horizontally_overlaps:
            # If aligned and overlapping, merge next_box into current_row_box
            current_row_box[0] = min(current_row_box[0], next_box[0])  # New x_min
            current_row_box[1] = min(current_row_box[1], next_box[1])  # New y_min
            current_row_box[2] = max(current_row_box[2], next_box[2])  # New x_max
            current_row_box[3] = max(current_row_box[3], next_box[3])  # New y_max
        else:
            # If not, the current row is complete. Add it to final rows.
            final_rows_aa.append(current_row_box)
            # Start a new row with the next_box.
            current_row_box = list(next_box)
            
    if current_row_box:  # Add the last processed row
        final_rows_aa.append(current_row_box)
        
    return final_rows_aa

def get_page_extents(all_axis_aligned_boxes):
    """
    Calculates the overall bounding box (min/max x and y coordinates)
    that encompasses all provided axis-aligned boxes.
    """
    if not all_axis_aligned_boxes:
        return 0, 0, 0, 0 # Default if no boxes
    
    all_x_coords = [b[0] for b in all_axis_aligned_boxes] + [b[2] for b in all_axis_aligned_boxes]
    all_y_coords = [b[1] for b in all_axis_aligned_boxes] + [b[3] for b in all_axis_aligned_boxes]
    
    if not all_x_coords: # Should not happen if all_axis_aligned_boxes is not empty
        return 0, 0, 0, 0

    return min(all_x_coords), max(all_x_coords), min(all_y_coords), max(all_y_coords)

def assign_box_to_column_by_overlap(box_aa, columns):
    """
    Assigns an axis-aligned box to a column based on maximum horizontal overlap.
    The box must overlap with the assigned column by more than 50% of its own width.

    Args:
        box_aa ([xmin, ymin, xmax, ymax]): The axis-aligned box to assign.
        columns (list of tuples): List of columns, where each column is (col_xmin, col_xmax).

    Returns:
        int or None: Index of the assigned column, or None if no suitable assignment.
    """
    if not columns or not box_aa: return None
    
    max_overlap_width = -1.0 # Start with negative to ensure any positive overlap is chosen
    assigned_col_idx = None
    box_width = box_aa[2] - box_aa[0]
    
    if box_width <= 1e-6: return None # Cannot assign zero-width box by overlap ratio

    for i, (col_xmin, col_xmax) in enumerate(columns):
        overlap_start = max(box_aa[0], col_xmin)
        overlap_end = min(box_aa[2], col_xmax)
        overlap_width = overlap_end - overlap_start
        
        if overlap_width > max_overlap_width:
            max_overlap_width = overlap_width
            assigned_col_idx = i
            
    # Ensure a significant part of the box overlaps with the assigned column
    if assigned_col_idx is not None and max_overlap_width > 0:
        # Box must overlap > 50% of its own width with the column to be confidently assigned
        if (max_overlap_width / box_width) > 0.5: 
            return assigned_col_idx
    return None


# --- Column Identification Methods ---
def identify_columns_simple(all_axis_aligned_boxes, page_min_x, page_max_x,
                            smooth_window=5, gap_threshold_factor=0.05, 
                            min_col_width_heuristic=30):
    """
    Simplified column identification using a horizontal projection profile of text boxes.
    It looks for gaps in the projection to determine column boundaries.

    Args:
        all_axis_aligned_boxes (list): List of all AA boxes on the page.
        page_min_x (float): Minimum x-coordinate of content on the page.
        page_max_x (float): Maximum x-coordinate of content on the page.
        smooth_window (int): Window size for smoothing the projection profile.
        gap_threshold_factor (float): Factor of max projection value to define a gap.
        min_col_width_heuristic (int): Minimum pixel width for a region to be considered a column.

    Returns:
        list: A list of tuples, where each tuple is (col_xmin, col_xmax) for an identified column.
              Returns a single page-wide column if no distinct columns are found.
    """
    if not all_axis_aligned_boxes:
        return []

    profile_start_x = math.floor(page_min_x)
    profile_end_x = math.ceil(page_max_x)
    profile_len = profile_end_x - profile_start_x

    if profile_len <= 0: # If page width is zero or negative
        return [(page_min_x, page_max_x)] if all_axis_aligned_boxes else []

    # Create a horizontal projection profile: count boxes covering each x-coordinate
    projection = np.zeros(profile_len, dtype=int)
    for xmin_box, _, xmax_box, _ in all_axis_aligned_boxes:
        start_idx = max(0, math.floor(xmin_box) - profile_start_x)
        end_idx = min(profile_len, math.ceil(xmax_box) - profile_start_x)
        if start_idx < end_idx : # Ensure valid range
            projection[start_idx:end_idx] += 1
    
    if np.sum(projection) == 0: # No text projected
        return [(page_min_x, page_max_x)] if all_axis_aligned_boxes else []

    # Smooth the projection profile to reduce noise
    projection_smooth = projection
    if smooth_window > 1 and smooth_window < len(projection):
        try:
            projection_smooth = np.convolve(projection, np.ones(smooth_window)/smooth_window, mode='same')
        except ValueError: # Can happen if smooth_window is too large for a small projection
            projection_smooth = projection # Use unsmoothed if convolution fails
            
    max_proj_val = np.max(projection_smooth)
    if max_proj_val == 0: # Should be caught by sum check, but for safety
        return [(page_min_x, page_max_x)] if all_axis_aligned_boxes else []

    # Set a threshold to identify gaps (areas with low projection values)
    threshold = max_proj_val * gap_threshold_factor

    columns = []
    in_column = False
    current_col_start_coord = -1

    # Iterate through the smoothed projection profile
    for i in range(profile_len):
        x_coord_abs = profile_start_x + i # Absolute x-coordinate on the page
        if projection_smooth[i] > threshold: # Current x-slice is part of text (above threshold)
            if not in_column: # Start of a new column segment
                in_column = True
                current_col_start_coord = x_coord_abs
        else: # Current x-slice is a gap (below threshold)
            if in_column: # End of a column segment
                in_column = False
                col_end_coord = x_coord_abs # Gap starts at x_coord_abs, so column ended before it
                if (col_end_coord - current_col_start_coord) >= min_col_width_heuristic:
                    columns.append((current_col_start_coord, col_end_coord))
                current_col_start_coord = -1 # Reset
    
    # If the profile ends while still in a column segment
    if in_column:
        col_end_coord = profile_start_x + profile_len # End of page content
        if (col_end_coord - current_col_start_coord) >= min_col_width_heuristic:
            columns.append((current_col_start_coord, col_end_coord))

    if not columns and all_axis_aligned_boxes: # If no columns found, assume one page-wide column
        return [(page_min_x, page_max_x)]
    
    # Post-process: Merge very close or slightly overlapping columns
    if len(columns) > 1:
        columns.sort(key=lambda c: c[0]) # Ensure sorted by x_min
        merged_cols = [list(columns[0])]
        for i in range(1, len(columns)):
            prev_col_xmin, prev_col_xmax = merged_cols[-1]
            curr_col_xmin, curr_col_xmax = columns[i]
            # If current column starts before or very close to where previous one ended, merge them
            # Allow a small gap (quarter of min_col_width_heuristic) for merging
            if curr_col_xmin < prev_col_xmax + (min_col_width_heuristic / 4.0): 
                merged_cols[-1][1] = max(prev_col_xmax, curr_col_xmax) # Extend current merged column
            else:
                merged_cols.append(list(columns[i]))
        # Ensure merged columns are still wide enough
        columns = [tuple(c) for c in merged_cols if (c[1]-c[0]) >= min_col_width_heuristic] 
            
    return columns if columns else [(page_min_x, page_max_x)]


def identify_columns_by_dynamic_years(ocr_results, page_min_x, page_max_x,
                                      note_keyword="Note", expected_col_count=4,
                                      year_is_4_digits=True, min_year=1990, max_year=2050,
                                      vertical_alignment_tolerance_factor=0.75, # % of Note height
                                      min_col_width_ratio_of_note=0.3):
    """
    Identifies columns based on a 'Note' keyword and dynamically found year headers
    (e.g., "2023", "2024") appearing on roughly the same vertical line as 'Note'.
    This is tailored for financial statements or similar tabular data.

    Args:
        ocr_results (list): List of (polygon_box, text_content) tuples from OCR.
        page_min_x (float): Min x-coordinate of page content.
        page_max_x (float): Max x-coordinate of page content.
        note_keyword (str): The keyword indicating the 'Note' column header.
        expected_col_count (int): The number of columns expected (e.g., Desc, Note, Year1, Year2).
        year_is_4_digits (bool): Whether to look for 4-digit numbers as years.
        min_year, max_year (int): Range for valid year numbers.
        vertical_alignment_tolerance_factor (float): Multiplier for Note box height to define
                                                     vertical alignment tolerance for year headers.
        min_col_width_ratio_of_note (float): Minimum width of a derived column, as a ratio
                                             of the 'Note' keyword box's width.
    Returns:
        list or None: List of (col_xmin, col_xmax) tuples for identified columns,
                      or None if key elements are not found or criteria not met.
    """
    note_header_aa = None
    potential_headers_on_note_line = [] # Store boxes that are vertically aligned with 'Note'

    # First, find the 'Note' keyword box. Prefer the topmost if multiple exist.
    for poly_box, (text, _) in ocr_results: # Assuming ocr_results are (box, (text, score))
        text_content = text # text is already the string
        aa_box = convert_to_axis_aligned(poly_box)
        if not aa_box: continue
        clean_text = text_content.strip()
        if clean_text == note_keyword:
            if note_header_aa is None or aa_box[1] < note_header_aa[1]: # Take the highest 'Note'
                note_header_aa = aa_box
    
    if not note_header_aa:
        print(f"Hint: Keyword '{note_keyword}' for Note column header not found. Dynamic year finding cannot proceed.")
        return None

    # Define vertical alignment parameters based on the 'Note' box
    note_y_min, note_y_max = note_header_aa[1], note_header_aa[3]
    note_height = note_y_max - note_y_min if note_y_max > note_y_min else 20.0 # Default height if zero
    vertical_tolerance = note_height * vertical_alignment_tolerance_factor
    note_y_center = (note_y_min + note_y_max) / 2.0
    
    note_box_width = note_header_aa[2] - note_header_aa[0]
    min_sensible_col_width = note_box_width * min_col_width_ratio_of_note \
                             if note_box_width > 0 else 20.0 # Default min width

    # Find all text boxes that are vertically aligned with the 'Note' keyword box
    for poly_box, (text, _) in ocr_results:
        text_content = text
        aa_box = convert_to_axis_aligned(poly_box)
        if not aa_box: continue
        box_y_center = (aa_box[1] + aa_box[3]) / 2.0
        if abs(box_y_center - note_y_center) < vertical_tolerance: # Check vertical alignment
            potential_headers_on_note_line.append({'text': text_content.strip(), 'box_aa': aa_box})

    # From these aligned boxes, identify potential year headers to the right of 'Note'
    year_candidates_on_line = []
    for header_info in potential_headers_on_note_line:
        text_val, box_aa = header_info['text'], header_info['box_aa']
        if box_aa[0] > note_header_aa[2]: # Must be to the right of the "Note" box's x_max
            is_year = False
            if year_is_4_digits and len(text_val) == 4 and text_val.isdigit():
                year_num = int(text_val)
                if min_year <= year_num <= max_year:
                    is_year = True
            if is_year:
                year_candidates_on_line.append(box_aa)

    # We expect at least two year columns (e.g., current year, previous year)
    if len(year_candidates_on_line) < 2: # For a 4-column layout (Desc, Note, Year1, Year2)
        print(f"Hint: Found {len(year_candidates_on_line)} year-like headers to the right of '{note_keyword}' "
              f"on the same line. Expected 2 for the year columns. Dynamic method failed.")
        return None

    # Sort year candidates by their x-coordinate to get Year1 and Year2 in order
    year_candidates_on_line.sort(key=lambda b: b[0])
    year1_header_aa = year_candidates_on_line[0]
    year2_header_aa = year_candidates_on_line[1]

    # Define column boundaries based on these key header boxes
    # Col 0 (Description): From page start to start of 'Note' column
    # Col 1 (Note): Defined by the 'Note' keyword box, ends at gutter before Year1
    # Col 2 (Year1): Defined by Year1 header, ends at gutter before Year2
    # Col 3 (Year2): Defined by Year2 header, ends at page end
    
    # Gutter midpoints are used to separate columns
    s0_page_start = page_min_x
    s1_note_col_start = note_header_aa[0]
    # End of Note col / Start of Year1 col is midpoint between Note_x_max and Year1_x_min
    s2_note_col_end_year1_start = (note_header_aa[2] + year1_header_aa[0]) / 2.0 \
                                   if note_header_aa[2] < year1_header_aa[0] else year1_header_aa[0]
    s2_note_col_end_year1_start = max(s1_note_col_start, s2_note_col_end_year1_start) # Ensure progression

    # End of Year1 col / Start of Year2 col is midpoint
    s3_year1_col_end_year2_start = (year1_header_aa[2] + year2_header_aa[0]) / 2.0 \
                                   if year1_header_aa[2] < year2_header_aa[0] else year2_header_aa[0]
    s3_year1_col_end_year2_start = max(s2_note_col_end_year1_start, s3_year1_col_end_year2_start) # Ensure progression
    
    s4_page_end = page_max_x

    # Tentative column definitions
    temp_cols = []
    # Col 0: Description (everything before 'Note' column starts)
    if s1_note_col_start > s0_page_start + min_sensible_col_width:
        temp_cols.append((s0_page_start, s1_note_col_start))
    else: # If desc column is too small or note starts at page edge, adjust.
        # This case implies the first column might be the 'Note' column itself or is missing.
        # For now, we'll assume the structure holds and refine later.
        temp_cols.append((s0_page_start, s1_note_col_start))


    # Col 1: Note
    if s2_note_col_end_year1_start > s1_note_col_start + min_sensible_col_width:
        temp_cols.append((s1_note_col_start, s2_note_col_end_year1_start))
    else: # Note column is too small
        temp_cols.append((s1_note_col_start, s1_note_col_start + min_sensible_col_width)) # Give it min width
        s2_note_col_end_year1_start = s1_note_col_start + min_sensible_col_width # Adjust next start

    # Col 2: Year 1
    if s3_year1_col_end_year2_start > s2_note_col_end_year1_start + min_sensible_col_width:
        temp_cols.append((s2_note_col_end_year1_start, s3_year1_col_end_year2_start))
    else:
        temp_cols.append((s2_note_col_end_year1_start, s2_note_col_end_year1_start + min_sensible_col_width))
        s3_year1_col_end_year2_start = s2_note_col_end_year1_start + min_sensible_col_width

    # Col 3: Year 2
    if s4_page_end > s3_year1_col_end_year2_start + min_sensible_col_width:
        temp_cols.append((s3_year1_col_end_year2_start, s4_page_end))
    else: # Year 2 column too small, or page ends abruptly
        temp_cols.append((s3_year1_col_end_year2_start, s3_year1_col_end_year2_start + min_sensible_col_width))


    # Final check to ensure columns are distinct, ordered, and meet min width.
    # This refinement step is crucial.
    final_columns = []
    last_x_end = page_min_x
    
    # If we expect 4 columns, ensure we have roughly that structure
    # The logic above might generate overlapping or too narrow columns if keywords are very close.
    # We need to define the 4 columns based on the identified separators s0, s1, s2, s3, s4
    
    # Column 1 (Description): s0 to s1 (start of Note keyword)
    # Column 2 (Note): s1 to s2 (midpoint before Year1 keyword)
    # Column 3 (Year1): s2 to s3 (midpoint before Year2 keyword)
    # Column 4 (Year2): s3 to s4 (page end)

    candidate_boundaries = sorted(list(set([s0_page_start, s1_note_col_start, s2_note_col_end_year1_start, s3_year1_col_end_year2_start, s4_page_end])))
    
    # Ensure boundaries are progressive
    progressive_boundaries = [candidate_boundaries[0]]
    for b in candidate_boundaries[1:]:
        if b > progressive_boundaries[-1] + min_sensible_col_width / 2.0: # Ensure some separation
             progressive_boundaries.append(b)
        elif b > progressive_boundaries[-1]: # If very close but still progressive, take the new one to avoid overlap
            progressive_boundaries[-1] = b


    if len(progressive_boundaries) >= expected_col_count : # Need at least N+1 boundaries for N columns
        for i in range(len(progressive_boundaries) -1):
            col_x1 = progressive_boundaries[i]
            col_x2 = progressive_boundaries[i+1]
            if col_x2 > col_x1 + min_sensible_col_width / 2.0: # Check width again
                 final_columns.append((col_x1, col_x2))
            # if len(final_columns) == expected_col_count: break # Stop if we have enough

    # If after this, we don't have the expected number, this method might not be suitable.
    if len(final_columns) >= expected_col_count: # Allow more if page naturally has more divisions
        # If we got more than expected, try to take the most plausible ones or the first 'expected_col_count'
        # For now, we'll take up to expected_col_count, or all if fewer.
        # This part might need more sophisticated logic if more than expected_col_count are found.
        # For a 4-column target:
        if len(final_columns) > expected_col_count:
             # Heuristic: if we have 5 columns from 4 expected, maybe the first "description"
             # column was split. Try to merge the first two if they are both before 'Note'.
             # This is getting very specific. For now, let's just take the first `expected_col_count`.
             # Or, if 'Note' is the second column, take one before, 'Note', and two after.
             
             # Find index of 'Note' column (whose start is s1_note_col_start)
             note_col_index = -1
             for idx, (cx1, _) in enumerate(final_columns):
                 if abs(cx1 - s1_note_col_start) < min_sensible_col_width / 2.0 : # If it starts near 'Note'
                     note_col_index = idx
                     break
             
             if note_col_index != -1 and note_col_index > 0 and (note_col_index + (expected_col_count - 2)) < len(final_columns) :
                 # Try to construct: [col_before_note, note_col, year1_col, year2_col]
                 # This assumes 'Note' is the second of the four main columns.
                 start_idx_for_selection = note_col_index -1
                 end_idx_for_selection = note_col_index + (expected_col_count -1) # e.g. note_idx + 2 for 3 more cols
                 if end_idx_for_selection <= len(final_columns):
                     final_columns = final_columns[start_idx_for_selection : end_idx_for_selection]


        if len(final_columns) > expected_col_count: # If still too many, truncate
            final_columns = final_columns[:expected_col_count]


        print(f"Successfully identified {len(final_columns)} columns using '{note_keyword}' and dynamically found year headers.")
        return final_columns
    else:
        print(f"Warning: Keyword-based (dynamic years) column ID resulted in {len(final_columns)} columns, expected {expected_col_count}.")
        return None


# --- Main Orchestration Function for Layout Analysis ---

def process_document_layout_with_ocr(
    ocr_results_tuples, # Expects list of (polygon_box, (text_content, score))
    word_expansion_factor_global=0.1, row_v_overlap_ratio_global=0.3,
    word_expansion_factor_in_col=0.02, row_v_overlap_ratio_in_col=0.4,
    col_id_method='keywords_else_simple', # 'keywords_only', 'simple_only'
    col_note_keyword="Note", 
    # Parameters for simple_only or fallback:
    col_smooth_window=5, col_gap_thresh_factor=0.05, col_min_width_pixels=50
):
    """
    Processes OCR results to determine document layout: columns, spanning rows, and in-column rows.

    Args:
        ocr_results_tuples (list): List of (polygon_box, (text_content, score)) tuples.
        word_expansion_factor_global (float): Horizontal expansion for global row formation.
        row_v_overlap_ratio_global (float): Vertical overlap for global row formation.
        word_expansion_factor_in_col (float): Horizontal expansion for in-column row formation.
        row_v_overlap_ratio_in_col (float): Vertical overlap for in-column row formation.
        col_id_method (str): 'keywords_else_simple', 'keywords_only', or 'simple_only'.
        col_note_keyword (str): Keyword for the 'Note' column (for keyword-based method).
        col_smooth_window, col_gap_thresh_factor, col_min_width_pixels: Params for simple column ID.

    Returns:
        tuple: (final_full_length_rows_in_cols, spanning_row_boxes, columns)
               - final_full_length_rows_in_cols: List of AA boxes for rows within columns,
                 extended to full column width.
               - spanning_row_boxes: List of AA boxes for rows that span multiple columns.
               - columns: List of (col_xmin, col_xmax) tuples for identified columns.
    """
    if not ocr_results_tuples: 
        return [], [], []

    # Extract initial polygon boxes and also keep the text for keyword search
    initial_polygon_boxes = [item[0] for item in ocr_results_tuples]
    
    # Convert all initial polygon boxes to axis-aligned boxes for general use
    all_aa_boxes = [b for b in [convert_to_axis_aligned(pb) for pb in initial_polygon_boxes] 
                    if b and (b[2]-b[0]>1e-6) and (b[3]-b[1]>1e-6)] # Filter invalid/small
    if not all_aa_boxes: 
        return [], [], []
    
    page_min_x, page_max_x, _, _ = get_page_extents(all_aa_boxes)
    
    columns = None
    # Attempt column identification using the chosen method
    if col_id_method in ['keywords_else_simple', 'keywords_only']:
        columns = identify_columns_by_dynamic_years(
            ocr_results_tuples, page_min_x, page_max_x,
            note_keyword=col_note_keyword
            # Other dynamic_years params use defaults
        )
    
    if columns is None and col_id_method in ['keywords_else_simple', 'simple_only']:
        if col_id_method == 'keywords_else_simple':
            print("Hint: Keyword-based (dynamic years) column ID failed or not selected, "
                  "falling back to simple projection method.")
        columns = identify_columns_simple(
            all_aa_boxes, page_min_x, page_max_x, 
            col_smooth_window, col_gap_thresh_factor, col_min_width_pixels
        )
    elif columns is None and col_id_method == 'keywords_only':
        print("Error: Keyword-based (dynamic years) column ID failed and no fallback specified. "
              "Treating page as a single column.")
        columns = [(page_min_x, page_max_x)] # Default to a single page-wide column
    
    if not columns: # If all methods fail, default to single column
        print("Warning: All column identification methods failed. Defaulting to single page-wide column.")
        columns = [(page_min_x, page_max_x)]


    # --- Identify Spanning Rows vs. Non-Spanning (potentially in-column) Rows ---
    # First, form rows globally from all initial polygon boxes
    globally_formed_rows_aa = create_rows_from_boxes(
        initial_polygon_boxes, 
        word_expansion_factor_global, 
        row_v_overlap_ratio_global
    )

    spanning_row_boxes = [] # Rows that clearly span multiple identified columns
    non_spanning_global_rows_aa = [] # Rows that seem to belong to a single column or unassigned

    if columns and len(columns) > 1:
        for r_xmin, r_ymin, r_xmax, r_ymax in globally_formed_rows_aa:
            cols_touched_indices = set()
            for i, (col_xmin_bound, col_xmax_bound) in enumerate(columns):
                # Check for overlap: max of starts < min of ends
                if max(r_xmin, col_xmin_bound) < min(r_xmax, col_xmax_bound):
                    cols_touched_indices.add(i)
            
            if len(cols_touched_indices) > 1:
                spanning_row_boxes.append([r_xmin, r_ymin, r_xmax, r_ymax])
            else:
                non_spanning_global_rows_aa.append([r_xmin, r_ymin, r_xmax, r_ymax])
    else: # If only one column (or no columns identified), all global rows are non-spanning by definition
        non_spanning_global_rows_aa = list(globally_formed_rows_aa)
    
    # --- Refine In-Column Rows ---
    # Filter initial polygon boxes: remove those largely subsumed by spanning rows
    non_spanning_initial_polys = []
    for poly_box in initial_polygon_boxes:
        aa_box_temp = convert_to_axis_aligned(poly_box)
        if not aa_box_temp: continue

        box_area = (aa_box_temp[2]-aa_box_temp[0]) * (aa_box_temp[3]-aa_box_temp[1])
        if box_area <= 1e-6: continue # Skip zero-area boxes

        is_subsumed_by_spanning_row = False
        for sx1,sy1,sx2,sy2 in spanning_row_boxes:
            # Calculate overlap area
            overlap_x_start = max(aa_box_temp[0], sx1)
            overlap_x_end = min(aa_box_temp[2], sx2)
            overlap_y_start = max(aa_box_temp[1], sy1)
            overlap_y_end = min(aa_box_temp[3], sy2)
            
            overlap_width = max(0, overlap_x_end - overlap_x_start)
            overlap_height = max(0, overlap_y_end - overlap_y_start)
            overlap_area = overlap_width * overlap_height
            
            # If a significant portion (e.g., >70%) of the box is within a spanning row, consider it subsumed
            if overlap_area / box_area > 0.7: 
                is_subsumed_by_spanning_row = True
                break
        
        if not is_subsumed_by_spanning_row:
            non_spanning_initial_polys.append(poly_box)

    # Now, for each column, take the non-spanning polygon boxes assigned to it
    # and re-form rows within that column, then extend these rows to full column width.
    final_full_length_rows_in_cols = []
    if columns:
        for col_idx, (col_xmin_bound, col_xmax_bound) in enumerate(columns):
            # Get polygon boxes that primarily belong to this column
            # using assign_box_to_column_by_overlap which checks for >50% width overlap
            boxes_for_this_col_poly = []
            for pb in non_spanning_initial_polys:
                aa_pb = convert_to_axis_aligned(pb)
                if aa_pb and assign_box_to_column_by_overlap(aa_pb, [columns[col_idx]]) == 0: # 0 because we pass a list with one col
                    boxes_for_this_col_poly.append(pb)
            
            if boxes_for_this_col_poly:
                # Create rows specifically from words within this column
                rows_made_in_col_aa = create_rows_from_boxes(
                    boxes_for_this_col_poly,
                    word_expansion_factor_in_col, # Tighter expansion within a column
                    row_v_overlap_ratio_in_col
                )
                # Extend these in-column rows to the full width of the column
                for r_xmin,r_ymin,r_xmax,r_ymax in rows_made_in_col_aa:
                    if r_ymax > r_ymin + 1e-3: # Ensure row has some height
                        # The row's x-coordinates are now the column's boundaries
                        final_full_length_rows_in_cols.append([col_xmin_bound, r_ymin, col_xmax_bound, r_ymax])
    
    elif non_spanning_global_rows_aa: # Fallback if no columns were properly identified (single column mode)
        # Treat global non-spanning rows as the final rows, extending them to page width
        for r_xmin,r_ymin,r_xmax,r_ymax in non_spanning_global_rows_aa:
            if r_ymax > r_ymin + 1e-3:
                final_full_length_rows_in_cols.append([page_min_x, r_ymin, page_max_x, r_ymax])
                
    return final_full_length_rows_in_cols, spanning_row_boxes, columns