import re
import cv2
import spacy
import numpy as np
import pytesseract
from PIL import Image
import io
import imagehash
from io import BytesIO
from typing import Dict, List
from pdf2image import convert_from_bytes


def compute_image_hash(file_bytes: bytes, filename: str) -> str:
    """
    Compute perceptual hash of the FIRST PAGE of PDF/image.
    Works for both images and PDFs.
    """
    try:
        if filename.lower().endswith('.pdf'):
            from pdf2image import convert_from_bytes
            # Convert first page only
            pages = convert_from_bytes(file_bytes, dpi=100, first_page=1, last_page=1)
            img = pages[0]
        else:
            img = Image.open(io.BytesIO(file_bytes))

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return str(imagehash.average_hash(img))
    except Exception:
        return ""  # Return empty if failed

def compute_text_fingerprint(text: str) -> str:
    """
    Create a normalized fingerprint from raw text.
    Removes noise, keeps structure.
    """
    if not text:
        return ""
    # Keep only alphanumeric + spaces, lowercase
    clean = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    # Collapse whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Take first 300 chars (enough for uniqueness)
    return clean[:300]


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

    # Use PSM 6 (assume single uniform block of text) — often better for invoices
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(pil_image, config=custom_config, lang='eng')
    return text.strip()


nlp = spacy.load("en_core_web_sm")

def extract_invoice_fields(text: str) -> Dict[str, str]:
    # Clean text
    clean_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()

    # Process with spaCy
    doc = nlp(clean_text)

    fields = {
        'date': '',
        'vendor': '',
        'invoice_id': '',
        'tax': '',
        'total_amount': ''
    }

    # === 1. VENDOR: Use ORG entities + context filtering 
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            # Skip generic terms
            if not re.search(r'(?i)\b(total|amount|tax|invoice|bill|payment|due|balance)\b', ent.text):
                orgs.append(ent.text.strip())

    # Pick the most vendor-like ORG (longest, or first non-generic)
    if orgs:
        fields['vendor'] = max(orgs, key=len)  # or orgs[0]

    # === 2. DATE: Use DATE entities + fallback regex
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    if dates:
        fields['date'] = dates[0]
    else:
        # Fallback to your regex
        date_match = re.search(r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b', clean_text, re.IGNORECASE)
        if date_match:
            fields['date'] = date_match.group(1)

    # === 3. INVOICE ID: Smarter regex (case-insensitive, more flexible)
    inv_match = re.search(r'(?:invoice|inv)[\s#:]*([A-Za-z0-9_\-]{6,20})', clean_text, re.IGNORECASE)
    if inv_match:
        fields['invoice_id'] = inv_match.group(1)

    # === 4. TOTAL AMOUNT: Look near "total" but avoid false positives
    # Find all currency-like numbers
    amounts = re.findall(r'[\d,]+\.?\d{0,2}', clean_text)
    if amounts:
        # Look for "total" within 30 chars of a number
        for amount in reversed(amounts):  # Start from largest/most likely total
            pattern = rf'(?:total|amount\s+due|grand\s+total|balance\s+due)[^\d]{{0,30}}({re.escape(amount)})'
            if re.search(pattern, clean_text, re.IGNORECASE):
                fields['total_amount'] = amount
                break

    # === 5. TAX: Similar logic ===
    tax_match = re.search(r'(?:tax|vat|gst)[^\d]*([\d,]+\.?\d{0,2})', clean_text, re.IGNORECASE)
    if tax_match:
        fields['tax'] = tax_match.group(1)

    return fields

def extract_line_items(text: str) -> List[Dict[str, str]]:
    """
    Future: Use table detection (OpenCV + Tesseract per cell) or LayoutLM.
    For now: return empty.
    """
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



