# [Experiment] PDF Table Extraction using Python Libraries

This Jupyter Notebook demonstrates and compares how various Python libraries extract tables from PDF files.

## Features

- **Text Extraction (PyMuPDF):**
    - Extracts plain text line by line.
    - Extracts structured text blocks with bounding box information.
    - Extracts individual words with bounding box information.
- **Table Detection (PyMuPDF):**
    - Utilizes PyMuPDF's basic built-in table finder.
- **Table Extraction (Camelot):**
    - Extracts tables using both 'lattice' (for tables with clear grid lines) and 'stream' (for tables without clear lines) methods.
    - Saves extracted tables into an Excel file, with separate sheets for lattice and stream results.
- **Table Extraction (Tabula-py):**
    - Extracts tables using both lattice and stream methods.
    - Saves extracted tables into a separate Excel file.
- **Table Extraction (PyMuPDF4LLM):**
    - Saves extracted tables into a MD file.
- **Table Extraction (Unstructured):**
    - Unstructured is available in both free and paid versions.
    - The paid version offers higher accuracy, but requires a subscription and processes documents on remote servers over the internet.
    - The free version can be used offline, but is less accurate compared to the paid version. 

## Prerequisites

Before running the script, ensure you have the following installed:

1. **Python 3.x**
2. **Required Python Libraries:**
    - `PyMuPDF`
    - `camelot-py[cv]` (the `[cv]` part installs OpenCV dependencies needed by Camelot)
    - `tabula-py`
    - `pandas`
    - `openpyxl` (for writing to Excel files)
3. **External Dependencies:**
    - **Ghostscript:** Required by Camelot for processing PDFs. Make sure it's installed and added to your system's PATH.  
      [Camelot Ghostscript Installation Guide](https://camelot-py.readthedocs.io/en/master/user/install-deps.html#ghostscript)
    - **Java Development Kit (JDK):** Required by Tabula-py. Make sure it's installed and `java` is accessible from your system's PATH.

## Observations

- These libraries do not use OCR methods. To use them, the PDF must have extractable text (not just images of text).
- In the tested pages, some tables are present only as images. These cannot be extracted as text or tables by these libraries.
- Camelot generally gives good results but fails when the text is not extractable. There are also many false positives.