"""Structured JSON request logging for the FastAPI service.

Deliberately lives in the API layer: `src/` is the shared inference core that the
Streamlit app imports too, so it must not grow a web concern. Nothing in `src/`
imports this module.

Emits one JSON object per line on stdout -- what Fly.io (and any container log
shipper) expects. Stdlib `logging` + `json` only; no structlog, no
python-json-logger, no new runtime dependency.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from datetime import UTC, datetime

# Attributes the stdlib puts on every LogRecord. Anything else a caller attaches
# via `extra=` is merged into the JSON payload as a top-level field.
_RESERVED = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
    # uvicorn attaches an ANSI-coloured duplicate of its own message; it is
    # noise in a structured log.
    "color_message",
})

REQUEST_ID_HEADER = "x-request-id"


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC)
                          .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def _header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _client_ip(scope: dict) -> str | None:
    """Prefer proxy-supplied client IP -- Fly terminates TLS in front of us."""
    for name in (b"fly-client-ip", b"x-forwarded-for"):
        raw = _header(scope, name)
        if raw:
            return raw.split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else None


class RequestLoggingMiddleware:
    """Pure-ASGI request logger.

    Pure ASGI rather than BaseHTTPMiddleware on purpose: BaseHTTPMiddleware wraps
    every response in an anyio task pair, which interferes with streaming
    responses and adds measurable per-request overhead for no benefit here.

    Honours an inbound X-Request-ID (so a request keeps its id across a proxy)
    and otherwise mints one; echoes it on the response either way.
    """

    def __init__(self, app, logger_name: str = "api.request") -> None:
        self.app = app
        self.logger = logging.getLogger(logger_name)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _header(scope, REQUEST_ID_HEADER.encode()) or uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        seen = {"status": 500}

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                seen["status"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        def fields(status: int) -> dict:
            return {
                "request_id": request_id,
                "method": scope.get("method"),
                "path": scope.get("path"),
                "query": scope.get("query_string", b"").decode("latin-1") or None,
                "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "client_ip": _client_ip(scope),
            }

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Unhandled error: log with traceback, then let it propagate so the
            # server still returns its own 500.
            self.logger.exception("request failed", extra=fields(500))
            raise

        status = seen["status"]
        level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 else logging.INFO
        self.logger.log(level, "request", extra=fields(status))


def configure_logging(level: int = logging.INFO) -> None:
    """Route all logging through one JSON handler on stdout.

    Idempotent, and safe to call both at import time (for tests) and again from
    the lifespan (uvicorn installs its own handlers after our import, so the
    second call is what actually wins in the container).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # src.utils.get_logger() attaches its own plain-text StreamHandler (and a
    # FileHandler) to named loggers. Drop the stream handlers so records reach
    # root exactly once as JSON instead of being emitted twice in two formats;
    # keep any FileHandler so outputs/logs/ still works locally, but reformat it.
    for name in ("api", "serving", "uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        for existing in list(logger.handlers):
            if isinstance(existing, logging.FileHandler):
                existing.setFormatter(JsonFormatter())
            else:
                logger.removeHandler(existing)
        logger.propagate = True

    # Our middleware logs every request with more context than uvicorn's access
    # line, so silence the duplicate rather than emit both.
    logging.getLogger("uvicorn.access").disabled = True
