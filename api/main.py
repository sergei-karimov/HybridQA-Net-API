"""
FastAPI приложение HybridQA-Net.

Запуск:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import router
from utils.logger import setup_logger

logger = setup_logger("hybridqa.api")


# ================================================================= Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация и очистка ресурсов при запуске/остановке сервера."""
    logger.info("Запуск HybridQA-Net API...")

    # Инициализируем pipeline при старте
    try:
        from src.pipeline import HybridQANet
        app.state.pipeline = HybridQANet(config_path="configs/config.yaml")

        # Автозагрузка production чекпоинта
        production_ckpt = Path("models/production.pt")
        if production_ckpt.exists():
            app.state.pipeline.load_checkpoint(str(production_ckpt))
            logger.info(f"Production чекпоинт загружен: {production_ckpt}")
        else:
            logger.warning(f"Production чекпоинт не найден: {production_ckpt} — используются pretrained веса")

        logger.info("Pipeline успешно инициализирован.")
    except Exception as e:
        logger.error(f"Ошибка инициализации pipeline: {e}", exc_info=True)
        app.state.pipeline = None

    # Инициализируем YOLO+CLIP анализатор (Variant B)
    try:
        from src.yolo_clip_pipeline import YoloCLIPAnalyzer
        app.state.yolo_clip = YoloCLIPAnalyzer(
            yolo_model="yolo11n.pt",
            clip_image_model="sentence-transformers/clip-ViT-B-32",
            clip_text_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
            similarity_threshold=0.25,
            device=None,   # auto-detect GPU
        )
        logger.info("YoloCLIPAnalyzer успешно инициализирован.")
    except Exception as e:
        logger.error(f"Ошибка инициализации YoloCLIPAnalyzer: {e}", exc_info=True)
        app.state.yolo_clip = None

    yield   # ← Сервер работает

    logger.info("Остановка HybridQA-Net API...")
    if hasattr(app.state, "pipeline") and app.state.pipeline:
        # Освободить ресурсы GPU/CPU
        del app.state.pipeline
    if hasattr(app.state, "yolo_clip"):
        del app.state.yolo_clip


# ================================================================= App
app = FastAPI(
    title="HybridQA-Net API",
    description=(
        "Гибридная мультимодальная система контроля качества. "
        "Анализирует соответствие продуктов стандартам качества "
        "по изображениям и текстовым документам."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ----------------------------------------------------------------- CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # В продакшне укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------- Routes
app.include_router(router, prefix="/api/v1")

# ----------------------------------------------------------------- Static UI
_ui_dir = Path(__file__).parent.parent / "ui" / "web"
if _ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")


# ----------------------------------------------------------------- Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Необработанная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера", "detail": str(exc)},
    )


# ----------------------------------------------------------------- OpenAPI schema with Bearer auth button
_schema = get_openapi(
    title=app.title,
    version=app.version,
    description=app.description,
    routes=app.routes,
)
_schema.setdefault("components", {}).setdefault("securitySchemes", {})["BearerAuth"] = {
    "type": "http",
    "scheme": "bearer",
    "bearerFormat": "JWT",
}
for _path in _schema.get("paths", {}).values():
    for _op in _path.values():
        if isinstance(_op, dict):
            _op.setdefault("security", [{"BearerAuth": []}])
app.openapi_schema = _schema

# ----------------------------------------------------------------- Root
@app.get("/", tags=["system"], summary="Корневой эндпоинт")
async def root():
    return {
        "service": "HybridQA-Net",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
