import base64
import io
import math
import os
import cv2
import imutils
import numpy as np
import fitz
from pathlib import Path
from PIL import Image

# ================= CONFIG =================

DPI = 150
MAX_PIXELS = 1_900_000
MAX_DIM = 1600
JPEG_QUALITY = 75


# ================= PDF → IMAGE =================

def pdf_page_to_image(page, dpi=DPI) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))

# ================= IMAGE NORMALIZATION =================

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

# ================= ORIENTATION LOGIC =================

def orientation_score(gray: np.ndarray) -> float:
    """
    Measures horizontal text-line strength.
    Correct orientation → strong horizontal variance.
    """
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = np.sum(bw == 0, axis=1)
    return np.var(proj)


def top_bottom_ratio(gray: np.ndarray) -> float:
    """
    Invoices have more text at the bottom (tables).
    Upside-down pages invert this pattern.
    """
    h = gray.shape[0]
    top = gray[: h // 2]
    bottom = gray[h // 2 :]

    _, bw_top = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw_bottom = cv2.threshold(bottom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    top_text = np.sum(bw_top == 0)
    bottom_text = np.sum(bw_bottom == 0)

    if bottom_text == 0:
        return 0.0

    return top_text / bottom_text

def fix_orientation_projection_safe(img: Image.Image) -> Image.Image:
    img_np = np.array(img)

    scores = {}
    for angle in (0, 90, 180, 270):
        rotated = img_np if angle == 0 else imutils.rotate_bound(img_np, angle)
        gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
        scores[angle] = orientation_score(gray)

    base_score = scores[0]

    # 🚫 NEVER consider 180°
    candidate_angles = [0, 90, 270]
    best_angle = max(candidate_angles, key=lambda a: scores[a])

    # Weak signal → keep original
    if scores[best_angle] < base_score * 1.35:
        return img

    # Sideways must be VERY strong
    if best_angle in (90, 270):
        if scores[best_angle] < base_score * 1.5:
            return img

    if best_angle == 0:
        return img

    rotated = imutils.rotate_bound(img_np, best_angle)
    return Image.fromarray(rotated)


def fix_small_skew_hough(img: Image.Image) -> Image.Image:
    """
    Only fixes small skew (±10°).
    NEVER rotates 90° or 180°.
    """
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

    # Only correct very small skew
    if abs(median_angle) < 0.5 or abs(median_angle) > 10:
        return img

    rotated = imutils.rotate_bound(img_np, median_angle)
    return Image.fromarray(rotated)

# ================= BASE64 (OPTIONAL) =================

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()

# load pdf take page and make image with fix_orientation_projection_safe and fix_small_skew_hough
# pdfs = list(Path("pdfs").glob("*.pdf"))
# for pdf_path in pdfs:
#     print(f"\n📄 Processing {pdf_path.name}")

#     doc = fitz.open(pdf_path)

#     for i in range(len(doc)):
#         print(f"  ├─ Page {i+1}")

#         img = pdf_page_to_image(doc[i])
#         img = normalize_image(img)
#         img = fix_orientation_projection_safe(img)
#         img = fix_small_skew_hough(img)

#         # save the img in output folder
    
#         os.makedirs("output", exist_ok=True)
#         img.save(Path("output") / f"{pdf_path.stem}_page_{i+1}.jpg")