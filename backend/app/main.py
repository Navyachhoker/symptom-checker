import time
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.db.database import engine, Base
from app.api.routes import router
from app.core.logging_config import setup_logging, set_request_id

setup_logging()
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")
    yield
    await engine.dispose()
    logger.info("DB engine disposed")


app = FastAPI(
    title="AI Medical Triage Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Short 8-char ID is plenty to grep for in logs and cheap to read
    # aloud/copy when debugging a specific report from a user.
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    request.state.request_id = request_id

    start    = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration}ms"
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."}
    )


app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "medical-triage-ai"}