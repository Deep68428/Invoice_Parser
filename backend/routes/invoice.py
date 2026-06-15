import asyncio
import pandas as pd
from loguru import logger
from fastapi.responses import JSONResponse
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.schemas.common import ErrorResponse
from backend.agents.invoice_agent import get_invoice_data, run_ocr
from backend.utils.image_utils import pdf_to_images, fix_orientation_projection_safe, fix_small_skew_hough
from backend.utils.table_utils import extract_valid_tables, html_table_to_df

# Lower DPI = smaller images = faster OCR; 150 is a good tradeoff for table extraction.
TABLE_EXTRACTION_DPI = 150


route = APIRouter()
log = logger.bind()


@route.post("/invoice/ocr", responses={500: {"model": ErrorResponse}})
async def invoice_ocr(file: UploadFile = File(...)):
    try:
        invoice_data = await get_invoice_data(file)
        return JSONResponse(invoice_data)
    except Exception as e:
        log.error(f"💥 Unexpected error in CV analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@route.post("/invoice/ocr-text", responses={500: {"model": ErrorResponse}})
async def invoice_ocr_text(file: UploadFile = File(...)):
    """
    Extract OCR text (LaTeX) from invoice PDF, plus extracted table as JSON.
    Returns LaTeX per page, full text, and table (columns + data) in JSON format.
    """
    try:
        pdf_bytes = await file.read()
        pages = pdf_to_images(pdf_bytes, dpi=TABLE_EXTRACTION_DPI)

        all_ocr_texts = []
        all_tables = []

        for page_num, img in enumerate(pages, 1):
            img = fix_orientation_projection_safe(img)
            img = fix_small_skew_hough(img)

            ocr_text = await asyncio.to_thread(run_ocr, img)
            all_ocr_texts.append({
                "page": page_num,
                "latex": ocr_text,
            })

            table_html = extract_valid_tables(ocr_text)
            if table_html:
                df = html_table_to_df(table_html)
                mask = (
                    df.applymap(
                        lambda v: (not pd.isna(v))
                        and not (isinstance(v, str) and (v or "").strip() == "")
                    ).sum(axis=1)
                    >= 4
                )
                df = df[mask]
                if not df.empty:
                    all_tables.append(df)

        full_text = "\n\n".join([p["latex"] for p in all_ocr_texts])

        table_data = None
        if all_tables:
            final_df = pd.concat(all_tables, ignore_index=True)
            final_df.columns = [str(c).strip().lower() for c in final_df.columns]
            records = final_df.fillna("").to_dict(orient="records")
            table_data = {
                "columns": list(final_df.columns),
                "data": records,
            }

        return JSONResponse({
            "success": True,
            "total_pages": len(pages),
            "pages": all_ocr_texts,
            "full_text": full_text,
            "table": table_data,
        })
    except Exception as e:
        log.error(f"💥 Unexpected error in OCR text extraction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
