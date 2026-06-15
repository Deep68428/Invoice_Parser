import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

from backend.core.config import get_settings
from backend.core.deps import get_ocr_lightson_model, get_processor
from backend.core.logging import setup_logging
from backend.middlewares.cors import setup_cors
from backend.middlewares.logging import logging_middleware
from backend.middlewares.request_id import request_id_middleware
from backend.middlewares.timing import timing_middleware
from backend.routes.invoice import route


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------------------------
    # Startup
    # -------------------------
    setup_logging()
    settings = get_settings()

    logger.info("🚀 Starting AI Agent App")
    logger.info(f"Base model: {settings.MODEL_NAME}")

    # Force-load heavy resources (fail fast)
    logger.info("🔄 Loading models...")
    get_processor()
    get_ocr_lightson_model()
    logger.info("✅ Models loaded")

    yield  # -------- App runs here -------- #

    # -------------------------
    # Shutdown
    # -------------------------
    logger.warning("🛑 Shutting down AI Agent App")


app = FastAPI(
    title="AI Agent App",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration via middleware helper
setup_cors(app)


app.middleware("http")(request_id_middleware)
app.middleware("http")(logging_middleware)
app.middleware("http")(timing_middleware)


app.include_router(route, prefix="/v1")


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "service": app.title,
        "version": app.version,
    }


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=8080,reload=True)
