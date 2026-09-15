"""
Centralized logging configuration.

Why this exists:
  Before this module, logging was set up ad hoc inside app/main.py
  (a bare logging.basicConfig call), and the agent modules
  (orchestrator.py, safety.py, specialists.py) had no way to tie a
  log line back to the specific /chat request that produced it.
  In a multi-turn, multi-agent system that's a real debugging cost —
  "the orchestrator logged a JSON parse failure" is much less useful
  than "request abc123 failed to parse the orchestrator's JSON on
  turn 4 of this session."

How it works:
  - `_request_id_ctx` is a contextvars.ContextVar, set once per
    incoming HTTP request (see app/main.py's log_requests
    middleware). ContextVars automatically propagate into every
    `await`ed coroutine running within that same request's asyncio
    task — which covers the full call chain from the FastAPI route
    handler, through `triage_graph.ainvoke()`, into the orchestrator,
    safety, and specialist modules — with zero extra plumbing
    required in any of those files.
  - `RequestIdFilter` reads that ContextVar and stamps it onto every
    LogRecord passing through a handler, so the request ID shows up
    in every log line without every module needing to pass it around
    explicitly.
  - Handlers are attached to the ROOT logger, so any module that
    just does `logging.getLogger(__name__)` (the standard pattern
    used throughout this codebase) picks this up automatically.

Usage:
  # once, at app startup (app/main.py):
  from app.core.logging_config import setup_logging, set_request_id
  setup_logging()

  # once per incoming request (middleware):
  set_request_id(str(uuid.uuid4())[:8])

  # anywhere else — no special import needed beyond stdlib logging:
  logger = logging.getLogger(__name__)
  logger.info("...")   # request_id is stamped in automatically
"""

import logging
import logging.handlers
import os
from contextvars import ContextVar

# ── Request ID context ─────────────────────────────────────────
# Default "-" so logs emitted outside a request (e.g. at startup,
# or in a background task) still format cleanly instead of erroring.
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> None:
    """Call once per incoming request, before any downstream logging."""
    _request_id_ctx.set(request_id)


def get_request_id() -> str:
    return _request_id_ctx.get()


class RequestIdFilter(logging.Filter):
    """Attaches the current request's ID to every LogRecord it sees."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True  # never actually filters records out, just annotates


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | req=%(request_id)s | "
    "%(name)s | %(message)s"
)


def setup_logging(
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
) -> None:
    """
    Configures the root logger with two handlers:
      - stdout (what Render/any platform log viewer picks up)
      - a rotating file (5 MB per file, 3 backups kept) so logs
        survive across restarts locally without growing unbounded

    Safe to call once at startup. Clears any handlers configured by
    a prior logging.basicConfig() call to avoid duplicate log lines.
    """
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    request_id_filter = RequestIdFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)

    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, log_file),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(request_id_filter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Quiet noisy third-party loggers so INFO-level app logs aren't
    # drowned out by every HTTP client call langchain/groq make.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)