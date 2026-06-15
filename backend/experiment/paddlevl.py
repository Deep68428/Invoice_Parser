import io
import sys
import math
import fitz
import cv2
import imutils
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCRVL
from box_detection import detect_tables_in_image
import pandas as pd


# ================= CONFIG =================

PDF_DIR = Path("img")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

DPI = 150
MAX_PIXELS = 1_900_000
MAX_DIM = 1600
# ================= ORIENTATION UTILS =================

def orientation_score(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sum(np.abs(sobelx)) / (np.sum(np.abs(sobely)) + 1e-6)


def fix_orientation_projection_safe(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    scores = {}

    for angle in (0, 90, 180, 270):
        rotated = img_np if angle == 0 else imutils.rotate_bound(img_np, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        scores[angle] = orientation_score(gray)

    base_score = scores[0]
    candidate_angles = [0, 90, 270]
    best_angle = max(candidate_angles, key=lambda a: scores[a])

    if scores[best_angle] < base_score * 1.35:
        return img

    if best_angle in (90, 270) and scores[best_angle] < base_score * 1.5:
        return img

    if best_angle == 0:
        return img

    rotated = imutils.rotate_bound(img_np, best_angle)
    return Image.fromarray(rotated)


def fix_small_skew_hough(img: Image.Image) -> Image.Image:
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return img

    angles = []
    for line in lines[:30]:
        rho, theta = line[0]
        angle = (theta - np.pi / 2) * 180 / np.pi
        angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)

    if abs(median_angle) < 0.5 or abs(median_angle) > 10:
        return img

    rotated = imutils.rotate_bound(img_np, median_angle)
    return Image.fromarray(rotated)

# ================= IMAGE UTILS =================

def pdf_page_to_image(page, dpi=DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def resize_by_pixel_budget(img: Image.Image) -> Image.Image:
    w, h = img.size
    pixels = w * h
    if pixels <= MAX_PIXELS:
        return img
    scale = math.sqrt(MAX_PIXELS / pixels)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def normalize_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
    return resize_by_pixel_budget(img)

# ================= OCR + TABLE EXTRACTION =================

pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://164.52.192.138:8080/v1",
    layout_threshold=8
)


def run_ocr_on_image(img: Image.Image, name_prefix: str):
    img_path = OUTPUT_DIR / f"{name_prefix}.jpg"
    img.save(img_path, "JPEG", quality=95)

    print(f"    🔍 OCR -> {img_path.name}")
    output = pipeline.predict(str(img_path))

    all_tables = []

    for res in output:
        print("res",res)
        parsed = res.get("parsing_res_list")

        for block in parsed:
            if block["block_label"] == "table":
                html = block["block_content"]
                dfs = pd.read_html(html)
                all_tables.extend(dfs)

    return all_tables

# ================= PDF PIPELINE =================

def process_pdfs():
    for pdf_path in sorted(PDF_DIR.glob("*.jpg")):
        print(f"\n📄 Processing {pdf_path.name}")
        doc = fitz.open(pdf_path)

        for i in range(len(doc)):
            print(f"  ├─ Page {i+1}")

            img = pdf_page_to_image(doc[i])
            img = normalize_image(img)
            img = fix_orientation_projection_safe(img)
            img = fix_small_skew_hough(img)
            imgs = detect_tables_in_image(img, threshold=0.7)
            for idx, img in enumerate(imgs):
                name_prefix = f"{pdf_path.stem}_page_{idx+1}"
                tables = run_ocr_on_image(img[0], name_prefix)

                for t_idy, df in enumerate(tables, 1):
                    csv_path = OUTPUT_DIR / f"{name_prefix}_table_{idx}_{t_idy}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"      ✅ Table saved -> {csv_path.name}")

# ================= MAIN =================

if __name__ == "__main__":
    process_pdfs()
