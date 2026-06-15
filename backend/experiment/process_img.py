import fitz
import requests
import io
import json
import numpy as np
from pathlib import Path
from PIL import Image

from img_process import (
    normalize_image,
    fix_orientation_projection_safe,
    fix_small_skew_hough,
    image_to_base64
)

# ================= CONFIG =================

PDF_DIR = Path("pdfs")

OLLAMA_URL = "http://164.52.192.138:11434/api/generate"
MODEL = "qwen2.5vl:7b"

DPI = 150
TIMEOUT = 300

BASE_TILE_HEIGHT = 900     # logical (header-safe split)
RETRY_TILE_HEIGHT = 450

HEADER_HEIGHT = 220        # persistent header height
OVERLAP = 80

# ================= PROMPT =================

PROMPT_BASE = """Extract ONLY the medicines/items table from the invoice image.
Return ONLY valid JSON. No text, no markdown.
Copy values EXACTLY as printed.
If a cell is empty or unreadable, return "".
Ignore all sections except the medicines table.

Output format:
{ "invoice_number":"","medicines": [ {} ] }

Rules:
- One table row = one object
- Keep all values as strings
- Preserve row order
- No extra keys
- No explanations
- Sometimes some data can be in two different lines; combine them
- Single characters may appear on new lines due to small column width combine that as well
- Do NOT miss any column or row

Expect below headers names might differ
- Sr No.
- Quantity (Qty)
- Particulars
- Pack
- HSN Code
- Batch No.
- Expiry Date
- MRP
- Rate
- Discount %
- IGST % / GST %
- Taxable Amount
- Net Amount
- Net Discount Amount
- Net Taxable Amount
- Round-off
- Grand Total

if any data not found you can use below logic to calculate that
Subtotal = Quantity × Rate
Discount Amount = Subtotal × Discount % ÷ 100
Taxable Value = Subtotal − Discount Amount
Tax Amount = Taxable Value × Tax % ÷ 100
Total Amount = Taxable Value + Tax Amount
""".strip()

# ================= IMAGE UTILS =================

def pdf_page_to_image(page, dpi=DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def split_with_header(img: Image.Image, header_height=HEADER_HEIGHT, overlap=OVERLAP):
    """
    Split image into 2 vertical tiles while keeping header in both.
    Max tiles returned = 2
    """
    w, h = img.size

    if h <= header_height + 50:
        return [img]

    header = img.crop((0, 0, w, header_height))
    body = img.crop((0, header_height, w, h))

    bw, bh = body.size
    mid = bh // 2

    tiles = []

    cuts = [
        (0, mid),
        (mid - overlap, bh)
    ]

    for y1, y2 in cuts:
        part = body.crop((0, max(0, y1), bw, min(bh, y2)))
        new_img = Image.new("RGB", (w, header_height + part.size[1]), (255, 255, 255))
        new_img.paste(header, (0, 0))
        new_img.paste(part, (0, header_height))
        tiles.append(new_img)

    return tiles


# ================= OLLAMA =================

def call_ollama(img: Image.Image):
    payload = {
        "model": MODEL,
        "prompt": PROMPT_BASE,
        "images": [image_to_base64(img)],
        "stream": False
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    raw = r.json().get("response", "")
    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise ValueError("Invalid JSON root")

    medicines = parsed.get("medicines", [])

    if not isinstance(medicines, list):
        raise ValueError("Invalid medicines list")

    return medicines


# ================= CONTROLLED OCR =================

def process_tile(tile: Image.Image):
    return call_ollama(tile)


def ocr_page(img: Image.Image):
    all_rows = []

    # ---------- LEVEL 0 ----------
    try:
        print("   🧠 Trying full image")
        return process_tile(img)

    except Exception as e:
        print("   ⚠️ Full image failed → split into 2")

    # ---------- LEVEL 1 ----------
    tiles_lvl1 = split_with_header(img)

    for idx, t1 in enumerate(tiles_lvl1, 1):
        try:
            print(f"   🧩 Tile {idx}/2")
            rows = process_tile(t1)
            all_rows.extend(rows)
            continue

        except Exception as e:
            print(f"   ⚠️ Tile {idx} failed → split again")

        # ---------- LEVEL 2 (only this tile) ----------
        tiles_lvl2 = split_with_header(t1)

        for jdx, t2 in enumerate(tiles_lvl2, 1):
            try:
                print(f"      🧩 Subtile {idx}.{jdx}")
                rows = process_tile(t2)
                all_rows.extend(rows)

            except Exception as e:
                raise RuntimeError("Tile failed after max split")

    return all_rows


# ================= MAIN =================

def process_pdfs():
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        print(f"\n📄 Processing {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)

            for i in range(len(doc)):
                print(f"  ├─ Page {i+1}")

                img = pdf_page_to_image(doc[i])
                img = normalize_image(img)
                img = fix_orientation_projection_safe(img)
                img = fix_small_skew_hough(img)

                rows = ocr_page(img)
                print(f"  ✅ Rows extracted: {len(rows)}")

        except Exception as e:
            print(f"  ❌ {e}")


if __name__ == "__main__":
    process_pdfs()
