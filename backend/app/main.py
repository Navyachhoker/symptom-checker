from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.database import engine, Base
from app.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables if they don't exist yet
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ Database tables ready")
    yield
    # Shutdown: dispose engine
    await engine.dispose()
    logger.info("🔌 DB engine disposed")


app = FastAPI(
    title="AI Medical Triage Assistant",
    description="Conversational triage agent powered by LangGraph + Groq",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "*",                       # tighten this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api
app.include_router(router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "medical-triage-ai"}