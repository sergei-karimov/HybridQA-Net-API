"""
Pydantic схемы для FastAPI endpoints.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ================================================================= Запросы
class AnalyzeRequest(BaseModel):
    """Запрос на анализ (JSON-body для текстовых данных)."""

    query: str = Field(
        default="Проверь соответствие продукта стандарту качества",
        description="Запрос для проверки соответствия.",
        max_length=2000,
    )
    standard_text: Optional[str] = Field(
        default=None,
        description="Текст стандарта качества (альтернатива загрузке файла).",
        max_length=50_000,
    )
    history: Optional[list[str]] = Field(
        default=None,
        description="История предыдущих запросов для контекста.",
        max_length=10,
    )
    defect_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог для бинаризации маски дефектов.",
    )
    use_cache: bool = Field(default=True, description="Использовать кэш результатов.")

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if v is not None:
            return [str(h)[:500] for h in v]
        return v


class BatchAnalyzeItem(BaseModel):
    """Один элемент пакетного запроса."""

    image_b64: str = Field(description="Base64-кодированное изображение.")
    query: str = Field(default="Проверь соответствие", max_length=2000)
    standard_text: Optional[str] = Field(default=None, max_length=50_000)


class BatchAnalyzeRequest(BaseModel):
    """Пакетный запрос на анализ."""

    items: list[BatchAnalyzeItem] = Field(
        description="Список элементов для анализа.",
        min_length=1,
        max_length=32,
    )
    use_cache: bool = Field(default=True)


# ================================================================= Ответы
class AnalyzeResponse(BaseModel):
    """Ответ с результатами анализа."""

    label: int = Field(description="0 = не соответствует, 1 = соответствует.")
    verdict: str = Field(description="Текстовый вердикт.")
    confidence: float = Field(description="Уверенность модели [0, 1].")
    defect_score: float = Field(description="Вероятность наличия дефекта [0, 1].")

    report: str = Field(description="Полный текстовый отчёт.")
    summary: str = Field(description="Краткое резюме.")
    defects: list[str] = Field(description="Список обнаруженных дефектов.")
    recommendations: list[str] = Field(description="Рекомендации.")

    attention_map_b64: Optional[str] = Field(
        default=None,
        description="Изображение с тепловой картой в base64 (PNG).",
    )
    processing_time_s: float = Field(description="Время обработки в секундах.")
    cached: bool = Field(description="Результат был взят из кэша.")


class BatchAnalyzeResponse(BaseModel):
    """Ответ на пакетный запрос."""

    results: list[AnalyzeResponse]
    total_processing_time_s: float


class ConditionItem(BaseModel):
    """Одно условие для проверки."""

    id: str
    query: str = Field(max_length=2000)
    type: Literal["must", "must_not"] = "must"


class ConditionResult(BaseModel):
    """Результат проверки одного условия."""

    id: str
    query: str
    type: str
    passed: bool
    confidence: float
    attention_map_b64: Optional[str] = None


class ConditionsResponse(BaseModel):
    """Агрегированный ответ на multi-condition запрос."""

    overall_pass: bool
    conditions_checked: int
    conditions_passed: int
    conditions_failed: int
    results: list[ConditionResult]
    total_processing_time_s: float


class HealthResponse(BaseModel):
    """Статус сервиса."""

    status: str
    version: str
    device: str
    cache_stats: dict


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""

    error: str
    detail: Optional[str] = None


# ================================================================= Auth
class TokenRequest(BaseModel):
    """Запрос токена аутентификации."""

    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class TokenResponse(BaseModel):
    """Ответ с токеном."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ================================================================= YOLO+CLIP schemas
class DetectionBox(BaseModel):
    region_name: str
    class_name: str
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    yolo_confidence: float = Field(ge=0.0, le=1.0)
    clip_similarity: float = Field(ge=-1.0, le=1.0)


class GridRegion(BaseModel):
    region_name: str
    clip_similarity: float = Field(ge=-1.0, le=1.0)
    crop_width: int
    crop_height: int


class SimilarityEntry(BaseModel):
    region_name: str
    similarity: float


class YoloCLIPResponse(BaseModel):
    label: int
    verdict: str
    confidence: float
    threshold: float
    yolo_detections: list[DetectionBox]
    grid_regions: list[GridRegion]
    best_region: str
    best_similarity: float
    all_similarities: list[SimilarityEntry]
    query: str
    normalized_query: str
    processing_time_s: float
    cached: bool


class YoloCLIPConditionResult(BaseModel):
    id: str
    query: str
    type: str
    passed: bool
    confidence: float
    best_region: str
    best_similarity: float
    yolo_detections_count: int


class YoloCLIPConditionsResponse(BaseModel):
    overall_pass: bool
    conditions_checked: int
    conditions_passed: int
    conditions_failed: int
    results: list[YoloCLIPConditionResult]
    total_processing_time_s: float
