# ============================================================
# HybridQA-Net Dockerfile
# Multi-stage build: builder + runtime
# ============================================================

# ---- Stage 1: Builder (установка зависимостей) -------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем только requirements для кэширования слоя
COPY requirements.txt .

# Устанавливаем зависимости в виртуальное окружение
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: Runtime (финальный образ) --------------------
FROM python:3.11-slim AS runtime

# Метаданные
LABEL maintainer="HybridQA-Net Team"
LABEL description="Hybrid Multimodal Quality Assurance System"
LABEL version="1.0.0"

WORKDIR /app

# Системные runtime-зависимости OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем Python-пакеты из builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY src/       ./src/
COPY api/       ./api/
COPY training/  ./training/
COPY utils/     ./utils/
COPY configs/   ./configs/

# Создаём директории для логов и чекпоинтов
RUN mkdir -p logs checkpoints

# Пользователь без привилегий
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false

# Порт API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Запуск
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
