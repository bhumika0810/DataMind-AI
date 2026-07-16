"""
DataMind AI — Backend Server
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from contextlib import asynccontextmanager
from pathlib import Path
import os

from routes.csv_routes import router as csv_router
from routes.ai_routes import router as ai_router
from routes.dataset_routes import router as dataset_router
from config import settings
from routes.database_routes import router as database_router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure uploads folder exists
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    print("✅ DataMind AI backend started")
    yield
    # Shutdown
    print("👋 DataMind AI backend stopped")


app = FastAPI(
    title="DataMind AI API",
    description="Upload CSVs, extract metadata, and query data with Gemini AI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (browser + file:// fallback via null origin) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ROUTES ──
app.include_router(csv_router, prefix="/api/csv",  tags=["CSV"])
app.include_router(ai_router,  prefix="/api/ai",   tags=["AI"])
app.include_router(dataset_router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(
    database_router,
    prefix="/api/database",
    tags=["Database"]
)

@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "running",
        "message": "DataMind AI API is live 🚀",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/app", include_in_schema=False)
async def app_redirect():
    return RedirectResponse("/datamind.html")


@app.get("/datamind.html", include_in_schema=False)
async def datamind_page():
    page = FRONTEND_DIR / "datamind.html"
    if not page.exists():
        return {"error": "Frontend not found", "path": str(page)}
    return FileResponse(page, media_type="text/html")


@app.get("/generated-charts/{filename}", include_in_schema=False)
async def generated_chart(filename: str):
    if "/" in filename or "\\" in filename or not filename.endswith(".png"):
        return {"error": "Invalid chart filename"}

    chart_path = Path(settings.DATA_DIR) / "charts" / filename
    if not chart_path.exists():
        return {"error": "Chart not found"}
    return FileResponse(chart_path, media_type="image/png")
