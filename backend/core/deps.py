from functools import lru_cache

import torch
from loguru import logger
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

from backend.core.config import get_settings

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

@lru_cache
def get_ocr_lightson_model():
    s = get_settings()
    logger.info(f"🤖 Loading OCR model: {s.MODEL_NAME}")
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        str(s.MODEL_NAME),
        torch_dtype=dtype
    ).to(device).eval()
    print("✅ model loaded")
    return model


@lru_cache
def get_processor():
    s = get_settings()
    logger.info(f"🤖 Loading OCR processor: {s.MODEL_NAME}")
    processor = LightOnOcrProcessor.from_pretrained(str(s.MODEL_NAME))
    print("✅ processor loaded")
    return processor

ocr_model = get_ocr_lightson_model()
processor = get_processor()
