import time

from fastapi import Request
from loguru import logger


async def logging_middleware(request: Request, call_next):
    start_time = time.time()

    # Get client IP address
    client_ip = request.headers.get(
        "X-Forwarded-For", request.client.host if request.client else "unknown"
    )
    # X-Forwarded-For can contain multiple IPs, take the first one
    client_ip = client_ip.split(",")[0].strip() if client_ip else "unknown"

    logger.info(f"➡️ {client_ip} | {request.method} {request.url.path}")

    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(
            f"🔥 Unhandled error on {client_ip} | {request.method} {request.url.path}"
        )
        raise e

    duration = (time.time() - start_time) * 1000

    logger.info(
        f"⬅️ {client_ip} | {request.method} {request.url.path} | "
        f"{response.status_code} | {duration:.2f} ms"
    )

    return response
