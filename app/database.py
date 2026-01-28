# database.py
import sqlite3
import os
import re
import json
from datetime import datetime
from typing import List
from PIL import Image
import io

DB_PATH = "invoices.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            fields TEXT NOT NULL,
            file_data BLOB,
            notes TEXT DEFAULT '',
            image_hash TEXT DEFAULT '',      -- NEW
            text_fingerprint TEXT DEFAULT ''  -- NEW
        )
    ''')

    # Safely add columns if missing
    for col in ['image_hash', 'text_fingerprint']:
        try:
            c.execute(f"ALTER TABLE invoices ADD COLUMN {col} TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Already exists

    conn.commit()
    conn.close()

def get_or_create_user(username: str) -> int:
    """Get user ID or create new user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = c.fetchone()

    if row:
        user_id = row[0]
    else:
        c.execute("INSERT INTO users (username) VALUES (?)", (username,))
        user_id = c.lastrowid

    conn.commit()
    conn.close()
    return user_id


def save_invoice(user_id: int, filename: str, file_content: bytes, result: dict, notes: str = ""):
    from ocr_pipeline import compute_image_hash, compute_text_fingerprint

    # Compute hashes
    img_hash = compute_image_hash(file_content, filename)
    text_fp = compute_text_fingerprint(result.get("raw_text", ""))

    stored_fields = result["fields"].copy()
    stored_fields.update({
        "page_count": result.get("page_count", 1),
        "raw_text_snippet": result.get("raw_text", "")[:500],
        "validation": result.get("validation", {}),
        "duplicates": []
    })

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    raw_text_to_save = result.get("raw_text", "")[:1000]
    c.execute('''
        INSERT INTO invoices (
            user_id, filename, upload_time, raw_text, fields, file_data, notes,
            image_hash, text_fingerprint
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        filename,
        datetime.now().isoformat(),
        raw_text_to_save,
        json.dumps(stored_fields),
        file_content,
        notes,
        img_hash,
        text_fp
    ))
    conn.commit()
    conn.close()


def get_user_invoices(user_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM invoices WHERE user_id = ?', (user_id,))
    rows = c.fetchall()
    conn.close()

    invoices = []
    for row in rows:
        # ✅ Convert Row to dict to enable .get()
        row_dict = dict(row)
        fields = json.loads(row_dict['fields'])

        invoice = {
            "id": row_dict["id"],
            "filename": row_dict["filename"],
            "upload_time": row_dict["upload_time"],
            "vendor": fields.get("vendor", ""),
            "invoice_id": fields.get("invoice_id", ""),
            "date": fields.get("date", ""),
            "total_amount": fields.get("total_amount", ""),
            "notes": row_dict.get("notes", ""),  # ✅ Now safe!
            "fields": fields
        }
        invoices.append(invoice)
    return invoices


def get_invoice_by_id(invoice_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM invoices WHERE id = ?', (invoice_id,))
    row = c.fetchone()
    conn.close()  # ← always close!

    if not row:
        return None

    # ✅ Use 'file_data', not 'image'
    file_bytes = row['file_data']  # ← correct column name

    return {
        'id': row['id'],
        'filename': row['filename'],
        'upload_time': row['upload_time'],
        'raw_text': row['raw_text'],
        'fields': json.loads(row['fields']),
        'file_data': file_bytes  # raw bytes (PDF or image)
    }

def delete_invoices_by_ids(user_id: int, invoice_ids: List[int]):
    """Delete selected invoices for user"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    placeholders = ','.join('?' * len(invoice_ids))
    c.execute(f'''
        DELETE FROM invoices
        WHERE id IN ({placeholders}) AND user_id = ?
    ''', (*invoice_ids, user_id))
    conn.commit()
    conn.close()


def parse_amount(s):
    try:
        return float(re.sub(r'[^\d.]', '', s)) if s else 0.0
    except:
        return 0.0

def find_duplicates(
    user_id: int,
    new_img_hash: str,
    new_text_fp: str,
    new_fields: dict
) -> list:
    """
    Find duplicates using 3-layer approach:
    1. Exact image hash match
    2. Text fingerprint match
    3. Field-based match (vendor + invoice_id + total)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Layer 1: Image hash
    if new_img_hash:
        c.execute('''
            SELECT * FROM invoices
            WHERE user_id = ? AND image_hash = ?
        ''', (user_id, new_img_hash))
        rows = c.fetchall()
        if rows:
            conn.close()
            return [dict(row) for row in rows]

    # Layer 2: Text fingerprint
    if new_text_fp:
        c.execute('''
            SELECT * FROM invoices
            WHERE user_id = ? AND text_fingerprint = ?
        ''', (user_id, new_text_fp))
        rows = c.fetchall()
        if rows:
            conn.close()
            return [dict(row) for row in rows]

    # Layer 3: Field-based (fallback)
    vendor = new_fields.get("vendor", "").lower()
    inv_id = new_fields.get("invoice_id", "")
    total = parse_amount(new_fields.get("total_amount", ""))

    if vendor and inv_id and total > 0:
        c.execute('''
            SELECT * FROM invoices
            WHERE user_id = ? AND fields LIKE ?
        ''', (user_id, f'%{inv_id}%'))
        rows = c.fetchall()
        for row in rows:
            fields = json.loads(row['fields'])
            if (fields.get("vendor", "").lower() == vendor and
                abs(parse_amount(fields.get("total_amount", "")) - total) / total <= 0.02):
                conn.close()
                return [dict(row)]

    conn.close()
    return []

