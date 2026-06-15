import logging
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def intercept_fastapi_logs():
    """Intercept FastAPI's standard logging and redirect to loguru."""
    import sys

    # Use loguru's intercept functionality
    # This patches the standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding loguru level
            level_map = {
                logging.DEBUG: "DEBUG",
                logging.INFO: "INFO",
                logging.WARNING: "WARNING",
                logging.ERROR: "ERROR",
                logging.CRITICAL: "CRITICAL",
            }
            level = level_map.get(record.levelno, "INFO")

            # Find the caller frame
            frame, depth = sys._getframe(6), 1
            while frame and frame.f_code.co_name in [
                "emit",
                "_log",
                "log",
                "info",
                "warning",
                "error",
                "critical",
            ]:
                frame = frame.f_back
                depth += 1

            # Log with loguru
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Set up interception for standard loggers
    handler = InterceptHandler()

    # Configure root logger to use our handler
    logging.getLogger().handlers = [handler]
    logging.getLogger().setLevel(logging.INFO)

    # Ensure uvicorn uses our logging
    for name in ["uvicorn", "uvicorn.access", "fastapi"]:
        logger_instance = logging.getLogger(name)
        logger_instance.handlers = [handler]
        logger_instance.propagate = False


def setup_logging():
    logger.remove()

    # Intercept FastAPI and related logs
    intercept_fastapi_logs()

    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        LOG_DIR / "app.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{process} | {thread} | "
            "{name}:{function}:{line} - {message}"
        ),
    )

    logger.info("✅ Logging initialized")
