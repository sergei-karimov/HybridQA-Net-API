"""
FastAPI маршруты HybridQA-Net API.
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from PIL import Image

from .middleware import (
    authenticate_user,
    create_access_token,
    get_current_user,
    rate_limit_middleware,
)
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    ConditionItem,
    ConditionsResponse,
    DetectionBox,
    ErrorResponse,
    GridRegion,
    HealthResponse,
    SimilarityEntry,
    TokenRequest,
    TokenResponse,
    YoloCLIPConditionResult,
    YoloCLIPConditionsResponse,
    YoloCLIPResponse,
)
from utils.logger import setup_logger

logger = setup_logger("hybridqa.api.routes")

router = APIRouter()

# Зависимость, которая инжектирует pipeline из app.state
def get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не инициализирована.",
        )
    return pipeline


# ================================================================= Auth
@router.post(
    "/auth/token",
    response_model=TokenResponse,
    summary="Получить токен аутентификации",
    tags=["auth"],
)
async def get_token(body: TokenRequest):
    """
    Аутентификация по логину/паролю. Возвращает JWT Bearer токен.
    """
    username = authenticate_user(body.username, body.password)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверные учётные данные.",
        )
    token = create_access_token(data={"sub": username})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=3600,
    )


# ================================================================= Health
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
    tags=["system"],
)
async def health_check(request: Request):
    """Возвращает статус сервиса, устройство и статистику кэша."""
    pipeline = getattr(request.app.state, "pipeline", None)
    cache_stats = pipeline.cache.stats() if pipeline else {"enabled": False}
    device = str(pipeline.device) if pipeline else "unknown"

    return HealthResponse(
        status="ok" if pipeline else "initializing",
        version="1.0.0",
        device=device,
        cache_stats=cache_stats,
    )


# ================================================================= Analyze
@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Анализ соответствия продукта стандарту",
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze(
    request: Request,
    image: UploadFile = File(description="Изображение продукта (JPEG/PNG/WebP)."),
    standard_doc: Optional[UploadFile] = File(
        default=None,
        description="Документ стандарта (PDF/TXT). Опционально.",
    ),
    query: str = Form(
        default="Проверь соответствие продукта стандарту качества",
        description="Запрос для проверки.",
    ),
    standard_text: Optional[str] = Form(
        default=None,
        description="Текст стандарта (альтернатива файлу).",
    ),
    defect_threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    use_cache: bool = Form(default=True),
    _rate_limit=Depends(rate_limit_middleware),
    current_user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
):
    """
    Выполнить анализ соответствия продукта стандарту качества.

    - **image**: Фото продукта (обязательно)
    - **standard_doc**: PDF или TXT с требованиями (опционально)
    - **query**: Конкретный вопрос или задача проверки
    """
    # Валидация изображения
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип изображения: {image.content_type}",
        )

    # Чтение изображения
    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Размер изображения превышает 20 МБ.",
        )

    # Чтение документа стандарта
    doc_content = None
    if standard_doc is not None:
        if standard_doc.content_type not in ("application/pdf", "text/plain"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Неподдерживаемый тип документа: {standard_doc.content_type}",
            )
        doc_content = await standard_doc.read()
    elif standard_text:
        doc_content = standard_text

    logger.info(
        f"Запрос от {current_user.get('sub', 'unknown')}: "
        f"image={image.filename}, query={query[:80]}"
    )

    try:
        result = pipeline.analyze(
            image=image_bytes,
            standard_doc=doc_content or "",
            query=query,
            defect_threshold=defect_threshold,
            use_cache=use_cache,
        )
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {str(e)}",
        )

    return AnalyzeResponse(
        label=result.label,
        verdict=result.verdict,
        confidence=result.confidence,
        defect_score=result.defect_score,
        report=result.report,
        summary=result.summary,
        defects=result.defects,
        recommendations=result.recommendations,
        attention_map_b64=result.attention_map_b64 or None,
        processing_time_s=result.processing_time_s,
        cached=result.cached,
    )


# ================================================================= Batch
@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    summary="Пакетный анализ",
    tags=["analysis"],
)
async def analyze_batch(
    request: Request,
    body: BatchAnalyzeRequest,
    _rate_limit=Depends(rate_limit_middleware),
    current_user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
):
    """
    Пакетный анализ нескольких изображений (до 32 элементов).
    Изображения передаются в base64.
    """
    start = time.perf_counter()

    items_prepared = []
    for item in body.items:
        try:
            image_bytes = base64.b64decode(item.image_b64)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Невалидный base64 в поле image_b64.",
            )
        items_prepared.append({
            "image": image_bytes,
            "standard_doc": item.standard_text or "",
            "query": item.query,
        })

    try:
        results = pipeline.analyze_batch(items_prepared, use_cache=body.use_cache)
    except Exception as e:
        logger.error(f"Ошибка пакетного анализа: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )

    elapsed = time.perf_counter() - start
    responses = [
        AnalyzeResponse(
            label=r.label,
            verdict=r.verdict,
            confidence=r.confidence,
            defect_score=r.defect_score,
            report=r.report,
            summary=r.summary,
            defects=r.defects,
            recommendations=r.recommendations,
            attention_map_b64=r.attention_map_b64 or None,
            processing_time_s=r.processing_time_s,
            cached=r.cached,
        )
        for r in results
    ]

    return BatchAnalyzeResponse(
        results=responses,
        total_processing_time_s=elapsed,
    )


# ================================================================= Conditions
@router.post(
    "/analyze/conditions",
    response_model=ConditionsResponse,
    summary="Проверка нескольких условий для одного изображения",
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_conditions(
    request: Request,
    image: UploadFile = File(description="Изображение продукта (JPEG/PNG/WebP)."),
    conditions_json: str = Form(description="JSON-массив условий [{id, query, type}]."),
    standard_text: Optional[str] = Form(default=None),
    use_cache: bool = Form(default=True),
    _rate_limit=Depends(rate_limit_middleware),
    current_user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
):
    """
    Проверить список условий соответствия для одного изображения.

    - **image**: Фото продукта (обязательно)
    - **conditions_json**: JSON-массив объектов `{id, query, type}` (max 20)
    - **standard_text**: Опциональный текст стандарта
    """
    # Валидация типа изображения
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип изображения: {image.content_type}",
        )

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Размер изображения превышает 20 МБ.",
        )

    # Парсинг и валидация условий
    try:
        raw_conditions = json.loads(conditions_json)
        if not isinstance(raw_conditions, list):
            raise ValueError("conditions_json должен быть массивом")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Невалидный conditions_json: {exc}",
        )

    if len(raw_conditions) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Максимальное число условий: 20.",
        )
    if len(raw_conditions) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="conditions_json не может быть пустым массивом.",
        )

    try:
        conditions = [ConditionItem(**c) for c in raw_conditions]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка валидации условий: {exc}",
        )

    logger.info(
        f"Запрос /conditions от {current_user.get('sub', 'unknown')}: "
        f"{len(conditions)} условий, image={image.filename}"
    )

    try:
        agg = pipeline.analyze_conditions(
            image=image_bytes,
            conditions=[c.model_dump() for c in conditions],
            standard_doc=standard_text or "",
            use_cache=use_cache,
        )
    except Exception as e:
        logger.error(f"Ошибка analyze_conditions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {str(e)}",
        )

    return ConditionsResponse(**agg)


# ================================================================= Cache
@router.delete(
    "/cache",
    summary="Очистить кэш",
    tags=["system"],
)
async def clear_cache(
    request: Request,
    current_user: dict = Depends(get_current_user),
    pipeline=Depends(get_pipeline),
):
    """Очистить весь кэш результатов. Требует аутентификации."""
    pipeline.cache.clear()
    return {"message": "Кэш очищен."}


# ================================================================= YOLO+CLIP
def get_yolo_clip(request: Request):
    analyzer = getattr(request.app.state, "yolo_clip", None)
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="YOLO+CLIP анализатор не инициализирован.",
        )
    return analyzer


@router.post(
    "/analyze/v2",
    response_model=YoloCLIPResponse,
    summary="YOLO+CLIP анализ соответствия (Variant B)",
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_v2(
    request: Request,
    image: UploadFile = File(description="Изображение продукта (JPEG/PNG/WebP)."),
    query: str = Form(
        default="Проверь соответствие продукта стандарту качества",
        description="Запрос на русском языке.",
    ),
    use_cache: bool = Form(default=True),
    _rate_limit=Depends(rate_limit_middleware),
    current_user: dict = Depends(get_current_user),
    analyzer=Depends(get_yolo_clip),
):
    """
    Анализ соответствия через YOLO-детекцию объектов + мультиязычный CLIP.
    Поддерживает русские запросы без предобучения.
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип изображения: {image.content_type}",
        )

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Размер изображения превышает 20 МБ.",
        )

    logger.info(
        f"[v2] Запрос от {current_user.get('sub', 'unknown')}: "
        f"image={image.filename}, query={query[:80]}"
    )

    try:
        result = analyzer.analyze(image_bytes, query, use_cache=use_cache)
    except Exception as e:
        logger.error(f"Ошибка analyze_v2: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {str(e)}",
        )

    return YoloCLIPResponse(
        label=result.label,
        verdict=result.verdict,
        confidence=result.confidence,
        threshold=result.threshold,
        yolo_detections=[DetectionBox(**d) for d in result.yolo_detections],
        grid_regions=[GridRegion(**g) for g in result.grid_regions],
        best_region=result.best_region,
        best_similarity=result.best_similarity,
        all_similarities=[SimilarityEntry(**s) for s in result.all_similarities],
        query=result.query,
        normalized_query=result.normalized_query,
        processing_time_s=result.processing_time_s,
        cached=result.cached,
    )


@router.post(
    "/analyze/v2/conditions",
    response_model=YoloCLIPConditionsResponse,
    summary="YOLO+CLIP проверка нескольких условий (Variant B)",
    tags=["analysis"],
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def analyze_v2_conditions(
    request: Request,
    image: UploadFile = File(description="Изображение продукта (JPEG/PNG/WebP)."),
    conditions_json: str = Form(description="JSON-массив условий [{id, query, type}]."),
    use_cache: bool = Form(default=True),
    _rate_limit=Depends(rate_limit_middleware),
    current_user: dict = Depends(get_current_user),
    analyzer=Depends(get_yolo_clip),
):
    """
    Проверить список условий через YOLO+CLIP для одного изображения.

    - **image**: Фото продукта (обязательно)
    - **conditions_json**: JSON-массив `[{id, query, type}]` (max 20)
    """
    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Неподдерживаемый тип изображения: {image.content_type}",
        )

    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Размер изображения превышает 20 МБ.",
        )

    try:
        raw_conditions = json.loads(conditions_json)
        if not isinstance(raw_conditions, list):
            raise ValueError("conditions_json должен быть массивом")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Невалидный conditions_json: {exc}",
        )

    if len(raw_conditions) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="conditions_json не может быть пустым массивом.",
        )
    if len(raw_conditions) > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Максимальное число условий: 20.",
        )

    try:
        conditions = [ConditionItem(**c) for c in raw_conditions]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Ошибка валидации условий: {exc}",
        )

    logger.info(
        f"[v2/conditions] Запрос от {current_user.get('sub', 'unknown')}: "
        f"{len(conditions)} условий, image={image.filename}"
    )

    try:
        agg = analyzer.analyze_conditions(
            image=image_bytes,
            conditions=[c.model_dump() for c in conditions],
            use_cache=use_cache,
        )
    except Exception as e:
        logger.error(f"Ошибка analyze_v2_conditions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка: {str(e)}",
        )

    return YoloCLIPConditionsResponse(
        overall_pass=agg["overall_pass"],
        conditions_checked=agg["conditions_checked"],
        conditions_passed=agg["conditions_passed"],
        conditions_failed=agg["conditions_failed"],
        results=[YoloCLIPConditionResult(**r) for r in agg["results"]],
        total_processing_time_s=agg["total_processing_time_s"],
    )
