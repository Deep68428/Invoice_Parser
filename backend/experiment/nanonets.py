import os
import re
import json
import base64
import tempfile
from io import BytesIO
from typing import List, Dict

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pdf2image import convert_from_path
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

from img_process import fix_orientation_projection_safe, fix_small_skew_hough

# ================= CONFIG =================

MODEL_PATH = "nanonets/Nanonets-OCR2-3B"
MAX_IMAGE_SIDE = 1024

app = FastAPI(title="Invoice Nanonets VL Extractor")

# ================= LOAD MODEL =================

print("🚀 Loading Nanonets OCR model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda",
    attn_implementation="eager"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("✅ Model loaded.")

# ================= VENDOR COLUMN MAP =================

VENDORS: Dict[str, List[str]] = {
    "aardee": ["sr_no","description_of_goods","hsn_sac","gst_rate","quantity","rate","per","disc","add_disc","exp","disc_amt","amount","batch_no"],
    "adlakha": ["sn","qty","free","itemname_and_packing","hsn_code","mrp","rate","batch","dis","sgst","cgst","amt"],
    "batrasons": ["sn","description_of_goods","hsn_sac_code","qty","expiry","list_price","discount","batch","price","amount"],
    "bhushan": ["sr","chln_no","particulars","hsn","pac","batcgh_no","exp","comp","qty","qty_dis","mrp","rate","dis","taxable","cgst","sgst","gst_value"],
    "comfort": ["s","qty","Mfr","pack","product_name","exp","batch","hsn","mrp","rate","dis","sgst","cgst","net_amount"],
    "gaba": ["s","qty","mfr","pack","product_name","mrp","exp","hsn","batch","rate","dis","sgst","cgst","amount","net"],
    "geess": ["sr","qty","particulars","pack","hsn_code","batch_no","exp","mrp","rate","dis","igst","net_amount"],
    "lenitive": ["s","qty","pack","product","batch","exp","hsn","mrp","rate","dis","igst","cgst","amount"],
    "luckey": ["sr","particulars","hsn_code","batch_no","exp","mrp","mrp","qty","rate","amount","gst","dis"],
    "marine": ["sno","item_name","hsn_code","qty","mrp","rate","sch","cd","cgst%","cgst_amt","sgst%","sgst_amt","amount"],
    "neelkanth": ["sr","qty","hsn_code","description","pack","batch_no","exp","mrp","rate","igst","amount","dis","scm","net_amt"],
    "plus": ["sn","item_name","menufacturer","batch_no","exp_date","hsn","qty","fqty","rate","mrp","amount","dis%","disc","gst","net_amount"],
    "ultra": ["sr","hsnsac","product_name","pkg_type","batch_no","mfg","exp","sample_qty","qty","rate","unit","mrp","igst","sgst","cgst","gst_amount","amount"]
}

# ================= IMAGE UTILS =================

def resize_for_vl(img: Image.Image, max_side=1024):
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w*s), int(h*s)))
    return img

# ================= PROMPT =================

def build_prompt(cols: List[str]) -> str:
    return f"""
Extract the medicines/items table AND the following invoice fields from the invoice image:

The table columns in this invoice are:
{cols}

Return ONLY valid minified JSON. No explanations.

Format EXACTLY:
{{
  "seller_name": "",
  "seller_gstin": "",
  "buyer_name": "",
  "buyer_gstin": "",
  "invoice_number": "",
  "invoice_date": "",
  "due_date": "",
  "rows": [...]
}}

Rules:
- Extract seller name, seller GSTIN, buyer name, buyer GSTIN, invoice number, invoice date, and due date from the invoice header/footer
- For the table: Use ONLY the above columns and same order.
- Each row must be an array of strings.
- Keep values EXACTLY as printed.
- Do NOT infer, calculate, normalize, or rename.
- Merge broken multi-line cells.
- If unreadable, use "".
- Extract all invoice fields even if the table is missing.
"""

# ================= HELPERS =================

def strip_code_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?", "", text, flags=re.I).strip()
    return text[text.find("{"):text.rfind("}")+1] if "{" in text else text

def normalize_output(raw, cols):
    if isinstance(raw, dict) and "rows" in raw:
        return raw

    rows = []
    for item in raw:
        row = [str(item.get(c, "")).strip() for c in cols]
        rows.append(row)

    return {"rows": rows}

# ================= CORE OCR =================

def extract_table_with_nanonets(img: Image.Image, prompt: str, cols: List[str]):
    img = resize_for_vl(img)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name
        img.save(path, "JPEG", quality=95)

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{path}"},
                {"type": "text", "text": prompt}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=12000, do_sample=False)

        gen_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        out_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        clean = strip_code_fences(out_text)
        raw = json.loads(clean)

        return normalize_output(raw, cols)

    finally:
        os.remove(path)

# ================= API =================

@app.get("/vendors")
def list_vendors():
    return {"vendors": list(VENDORS.keys())}

@app.post("/extract-invoice")
async def extract_invoice(vendor: str = Form(...), pdf: UploadFile = File(...)):
    if vendor not in VENDORS:
        raise HTTPException(400, "Invalid vendor")

    cols = VENDORS[vendor]
    prompt = build_prompt(cols)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, pdf.filename)
        with open(pdf_path, "wb") as f:
            f.write(await pdf.read())

        pages = convert_from_path(pdf_path, dpi=200)
        results = []

        for i, page in enumerate(pages, 1):
            page = fix_orientation_projection_safe(page.convert("RGB"))
            page = fix_small_skew_hough(page)

            data = extract_table_with_nanonets(page, prompt, cols)
            results.append({"page": i, "result": data})

    return JSONResponse({"vendor": vendor, "columns": cols, "pages": results})

# ================= RUN =================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("nanonets_invoice_api:app", host="0.0.0.0", port=8000, reload=False)
