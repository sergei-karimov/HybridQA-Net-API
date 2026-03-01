# HybridQA-Net — Память проекта

## Статус
Полная реализация создана. Pipeline A (HybridQANet) + Pipeline B (YOLO+CLIP). README и requirements.txt актуальны.
Web UI: `ui/web/index.html` (тёмная тема, canvas overlay, бары) — доступен на `/ui/`.
Rust TUI: `ui/tui/` (ratatui 0.28, grid table + bar chart). Последний коммит: d270d9a.

## Архитектура
- **Vision**: `src/vision/backbone.py` (ViT/EfficientNet через timm) + Grad-CAM++ (`gradcam.py`)
- **NLP**: `src/nlp/context_analyzer.py` (RuBERT/LaBSE через transformers)
- **Fusion**: `src/fusion/cross_attention.py` (двунаправленный cross-attention) + `decision_head.py`
- **Report**: `src/report/generator.py` (T5: `cointegrated/rut5-base`)
- **Pipeline**: `src/pipeline.py` — главный класс `HybridQANet`
- **API**: `api/main.py` FastAPI + JWT + rate limiting
- **Training**: `training/trainer.py` — AdamW + CosineAnnealingLR

## Ключевые модели (HuggingFace)
- Vision backbone: `vit_base_patch16_224` (timm)
- NLP: `DeepPavlov/rubert-base-cased`
- Report: `cointegrated/rut5-base`

## Запуск
```bash
pip install -r requirements.txt
python example_usage.py          # Пример (Pipeline A)
uvicorn api.main:app --port 8000 # API (оба pipeline)
docker-compose up --build        # Docker
```
- При старте API автоматически поднимает оба pipeline (HybridQANet + YoloCLIPAnalyzer)
- Первый запуск Pipeline B скачивает yolo11n.pt (~6MB) автоматически

## YOLO+CLIP Pipeline — Variant B (реализовано)
- `src/vision/detector.py`: YOLODetector (yolo11n.pt) + 3×3 grid = до 30 кропов
- `src/vision/clip_matcher.py`: CLIPMatcher — **два** SentenceTransformer:
  - `img_model`: `clip-ViT-B-32` → PIL изображения (CLIPModel)
  - `txt_model`: `clip-ViT-B-32-multilingual-v1` → только текст, поддерживает русский
  - ВАЖНО: `clip-ViT-B-32-multilingual-v1` — text-only, PIL передавать нельзя!
- `src/yolo_clip_pipeline.py`: YoloCLIPAnalyzer, analyze() + analyze_conditions()
  - `_normalize_query()`: убирает нормативные конструкции перед CLIP
    ("должен быть" → "", "необходимо" → "", "нужно" → "" и т.д.)
  - Причина: CLIP понимает описания, не предписания; "должен быть" снижает similarity
  - Ответ содержит `query` (оригинал) и `normalized_query` (что отправлено в CLIP)
  - `_apply_spatial_boost()`: реранкинг регионов по пространственным словам в запросе
    - Коэффициент ×1.15 для регионов, соответствующих направлению
    - "внизу/снизу/нижн" → bottom-*, "сверху/верхн" → top-*
    - "слева/левый" → *-left, "справа/правый" → *-right
    - "в центре/центральн" → center, top-center, bottom-center
    - Меняет только `best_region`; `confidence` и `all_similarities` — исходные CLIP-значения
    - Причина: без буста CLIP выбирает регион по максимуму схожести без учёта "внизу" и т.п.
    - ✓ Протестирован: внизу, слева, справа, сверху, в центре — все работают корректно
- Эндпоинты: `POST /api/v1/analyze/v2`, `POST /api/v1/analyze/v2/conditions`
- Порог: similarity_threshold=0.25, возвращается в ответе; кэш in-memory dict
- cv2: opencv-python (non-headless) конфликтует с libGL в WSL2 → удалить, оставить только headless
- `src/vision/__init__.py`: обёрнут в try/except — cv2/libGL ошибки не ломают импорт

## Multi-condition endpoint (реализовано)
- `api/schemas.py`: `ConditionItem`, `ConditionResult`, `ConditionsResponse`
- `src/pipeline.py`: метод `analyze_conditions(image, conditions, standard_doc, use_cache)`
- `api/routes.py`: `POST /api/v1/analyze/conditions` — JSON-массив условий (max 20), поле `type`: "must"|"must_not"
- `tools/generate_dataset.py`: полный редизайн — 2 зоны (A/B), 12 аннотаций на изображение
- Credentials API: admin/password123, user/user_password, readonly/readonly_pass

## Production модель
- Лучший чекпоинт: `models/production.pt` (val acc 81.53%, эпоха 10 unfrozen)
- Автозагрузка в `api/main.py` lifespan: если файл существует — грузится автоматически
- История тренировок:
  - Прогон 1: freeze_backbone=True, 50 эп., lr=1e-4 → val acc 79.86%
  - Прогон 2: freeze_backbone=False, 20 эп., lr=2e-5, batch=8, от чекпоинта эп.30 → val acc 81.53%
- Запуск API: `.venv/bin/python -m uvicorn api.main:app --port 8000` (через run_in_background)

## Датасет
- 300 изображений (240 train / 60 val) × 12 условий = 2880 / 720 аннотаций
- Генерация: `python tools/generate_dataset.py --count 300`
- Зона A: x=440–590, y=100–220 (правый верхний)
- Зона B: x=440–590, y=225–345 (правый центр)

## Rust TUI — детали
- `ui/tui/Cargo.toml`: reqwest с `rustls-tls` (не требует `libssl-dev`), без `default-features`
- `ui/tui/src/main.rs`: хоткеи `a`/`c`/`q` срабатывают только при фокусе на SubmitButton
  - При фокусе на ImagePath или Query все символы вводятся как текст
  - `Esc` всегда выходит независимо от фокуса
- Сборка: `cd ui/tui && cargo build` (требует `build-essential`)

## Важные детали реализации
- Forward hooks в backbone.py для извлечения промежуточных признаков
- GradCAM использует surrogate backward через embeddings.sum()
- Cache: LRU in-memory или Redis (переключается в config.yaml)
- Auth: demo users в middleware.py (DEMO_USERS dict)
- Все тензоры явно переносятся на self.device
- Запуск фоновых процессов: использовать `run_in_background=true` в Bash, НЕ `&` (exit code 144)
