import os
import fitz
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import imutils

# Import the required functions from img_process
from img_process import (
    pdf_page_to_image,
    normalize_image,
    orientation_score,
    top_bottom_ratio,
    fix_orientation_projection_safe,
    fix_small_skew_hough
)

def create_dataset_images():
    """
    Create images of all PDF pages and store them in dataset folder.
    Uses orientation correction functions from img_process.py
    """
    # Source directory for PDFs
    pdf_dir = Path("/home/ethics/Downloads/Odoo Images/pdfs/")
    # Destination directory for images
    dataset_dir = Path("dataset2")
    dataset_dir.mkdir(exist_ok=True)

    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    total_pages = 0

    for pdf_path in pdf_files:
        print(f"\n📄 Processing {pdf_path.name}")

        try:
            doc = fitz.open(pdf_path)

            for i in range(len(doc)):
                print(f"  ├─ Page {i+1}")

                # Convert PDF page to image
                img = pdf_page_to_image(doc[i])

                # Normalize image size
                img = normalize_image(img)

                # Fix orientation using projection-based method
                img = fix_orientation_projection_safe(img)

                # Fix small skew using Hough transform
                img = fix_small_skew_hough(img)

                # Save the processed image to dataset folder
                output_filename = f"{pdf_path.stem}_page_{i+1}.jpg"
                output_path = dataset_dir / output_filename
                img.save(output_path, "JPEG", quality=75)

                total_pages += 1

            doc.close()

        except Exception as e:
            print(f"  ❌ Error processing {pdf_path.name}: {str(e)}")
            continue

    print(f"\n✅ Dataset creation complete!")
    print(f"   Total pages processed: {total_pages}")
    print(f"   Images saved to: {dataset_dir.absolute()}")

if __name__ == "__main__":
    create_dataset_images()