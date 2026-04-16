"""HTTP request logging middleware (loguru)."""

from __future__ import annotations

import time
import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Bind ``request_id`` into loguru context, log one line per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex

        with logger.contextualize(request_id=req_id):
            t0 = time.monotonic()
            status_code = 500
            try:
                response = await call_next(request)
                status_code = response.status_code
                return response
            finally:
                duration_ms = (time.monotonic() - t0) * 1000
                logger.info(
                    "http_request  method={method} path={path} status={status} "
                    "duration_ms={duration:.2f} client={client}",
                    method=request.method,
                    path=request.url.path,
                    status=status_code,
                    duration=duration_ms,
                    client=request.client.host if request.client else None,
                )
