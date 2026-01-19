import re
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
from io import BytesIO
from typing import Dict, List
from pdf2image import convert_from_bytes



def preprocess_image(image):
    # Convert PIL to OpenCV format
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return threshold

def extract_text(image):
    processed_img = preprocess_image(image)
    pil_image = Image.fromarray(processed_img)
    custom_config = r'--oem 3 --psm 1'  # Fully automatic page segmentation
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text.strip()


def extract_invoice_fields(text: str) -> Dict[str, str]:
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    fields = {
        'date': '',
        'vendor': '',
        'invoice_id': '',
        'tax': '',
        'total_amount': ''
    }

    # 1. Date (DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD)
    date_match = re.search(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b', text, re.IGNORECASE)
    if date_match:
        fields['date'] = date_match.group(1)

    # 2. Invoice ID (alphanumeric code after "Invoice" or "INV")
    inv_match = re.search(r'(?:invoice|inv)[\s#:]*([A-Z0-9\-]{6,20})', text, re.IGNORECASE)
    if inv_match:
        fields['invoice_id'] = inv_match.group(1)

    # 3. Vendor (first capitalized words at document start)
    vendor_match = re.search(r'^([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})', text)
    if vendor_match:
        fields['vendor'] = vendor_match.group(1)

    # 4. Total amount (look for "total", "amount due", etc.)
    total_match = re.search(r'(?:total|amount\s+due|grand\s+total|balance\s+due)[^\d]*([\d,]+\.?\d{0,2})', text, re.IGNORECASE)
    if total_match:
        fields['total_amount'] = total_match.group(1)

    # 5. Tax (VAT, Tax, GST)
    tax_match = re.search(r'(?:tax|vat|gst)[^\d]*([\d,]+\.?\d{0,2})', text, re.IGNORECASE)
    if tax_match:
        fields['tax'] = tax_match.group(1)

    return fields


def extract_line_items(text: str) -> List[Dict[str, str]]:

    # To Enhance with table OCR or NLP later(future work)
    return []



def extract_text_from_file(file_content: bytes, filename: str) -> List[str]:

    pages_text = []

    if filename.lower().endswith('.pdf'):
        # Convert PDF to images (200 DPI for speed + quality)
        try:
            images = convert_from_bytes(file_content, dpi=200)
            for img in images:
                text = extract_text(img)
                pages_text.append(text)
        except Exception as e:
            raise ValueError(f"PDF processing failed: {str(e)}")
    else:
        # Handle image
        try:
            image = Image.open(io.BytesIO(file_content))
            text = extract_text(image)
            pages_text.append(text)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    return pages_text


def process_invoice_file(file_content: bytes, filename: str) -> Dict:
    pages_text = extract_text_from_file(file_content, filename)

    all_fields = []
    all_line_items = []
    full_raw_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
    
    for i, text in enumerate(pages_text):
        fields = extract_invoice_fields(text)
        line_items = extract_line_items(text)
        all_fields.append(fields)
        all_line_items.append(line_items)

    # For simplicity, return first page's fields (or merge logic later)
    main_fields = all_fields[0] if all_fields else {}

    return {
        "filename": filename,
        "raw_text": full_raw_text or "",  # ← never None
        "fields": main_fields,
        "page_count": len(pages_text)
    }


def create_download_link(text):
    buffer = BytesIO()
    buffer.write(text.encode('utf-8'))
    buffer.seek(0)
    return buffer


