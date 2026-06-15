from typing import Any


import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from loguru import logger

KEYWORDS = ["amount", "mrp", "hsn", "qty", "quantity"]


def _has_similar_keyword(text: str, keywords: list[str], threshold: int) -> bool:
    """
    Return True if any keyword is sufficiently similar to the text.

    Uses fuzzy matching to tolerate OCR noise/typos (e.g. 'quantlty', 'arnount').
    """
    if not text:
        return False
    text = text.lower()
    for k in keywords:
        # partial_ratio works well when the keyword is a small part
        # of a larger blob of extracted table text.
        score = fuzz.partial_ratio(k, text)
        if score >= threshold:
            logger.debug(
                "Fuzzy keyword match: keyword='{}' score={} threshold={}",
                k,
                score,
                threshold,
            )
            return True
    return False


def extract_valid_tables(html_text: str, *, threshold: int = 85, keywords: list[str] = KEYWORDS):
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table")
    logger.debug(
        "Scanning HTML for invoice tables: tables_found={} threshold={} keywords={}",
        len(tables),
        threshold,
        keywords,
    )

    for idx, table in enumerate[Any](tables):
        text = table.get_text(" ").lower()
        if _has_similar_keyword(text, keywords=keywords, threshold=threshold):
            logger.info("Matched invoice table: index={} of {}", idx, len(tables))
            return str(table)

    logger.warning(
        "No invoice table matched keywords: tables_found={} threshold={}",
        len(tables),
        threshold,
    )
    return None

def html_table_to_df(html: str):
    return pd.read_html(StringIO(html))[0]
