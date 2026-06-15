import asyncio
from typing import Any

import torch
import pandas as pd
from PIL import Image
from fastapi import HTTPException
from loguru import logger

from backend.core.deps import ocr_model, processor, device, dtype
from backend.utils.table_utils import extract_valid_tables, html_table_to_df
from backend.utils.image_utils import pdf_to_images, fix_orientation_projection_safe, fix_small_skew_hough

# Lower DPI for table extraction = smaller images = faster OCR.
TABLE_EXTRACTION_DPI = 150


def run_ocr(img: Image.Image) -> str:
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "url": img}],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        out = ocr_model.generate(**inputs, max_new_tokens=6000)

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(gen_ids, skip_special_tokens=True)


async def get_invoice_data(file):
    pdf_bytes = await file.read()
    pages = pdf_to_images(pdf_bytes, dpi=TABLE_EXTRACTION_DPI)

    all_tables = []

    for _, img in enumerate(pages, 1):
        img = fix_orientation_projection_safe(img)
        img = fix_small_skew_hough(img)

        ocr_text = await asyncio.to_thread(run_ocr, img)

        table_html = extract_valid_tables(ocr_text)
        if not table_html:
            raise HTTPException(status_code=404, detail=str("Valid table Not found"))
        df = html_table_to_df(table_html)
        # drop rows that have data in fewer than 4 columns (non-empty / non-NaN)
        mask = df.applymap(
            lambda v: (not pd.isna(v)) and (not (isinstance(v, str) and v.strip() == ""))
        ).sum(axis=1) >= 4
        dropped = len(df) - mask.sum()
        if dropped:
            logger.info("Dropping sparse rows: removed={} kept={}", dropped, mask.sum())
        df = df[mask]
        all_tables.append(df)

    if not all_tables:
        return HTTPException(status_code=404, detail=str("Table Not found"))

    final_df = pd.concat(all_tables, ignore_index=True)
    final_df.columns = [str(c).strip().lower() for c in final_df.columns]
    return {
        "success": True,
        "rows": len(final_df),
        "columns": list[str](final_df.columns),
        "data": final_df.fillna("").to_dict(orient="records")
    }
