import io
import cv2
import fitz
import imutils
import numpy as np
from PIL import Image


def orientation_score(gray: np.ndarray) -> float:
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    proj = np.sum(bw == 0, axis=1)
    return np.var(proj)


def top_bottom_ratio(gray: np.ndarray) -> float:
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


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)
    return images
