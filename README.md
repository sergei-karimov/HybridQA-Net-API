# HybridQA-Net

Гибридная мультимодальная система контроля качества. Поддерживает два независимых pipeline: классический (ViT + RuBERT) и нулевого-обучения (YOLO + CLIP).

## Архитектура

### Pipeline A — HybridQANet (классический, с дообучением)

```
Изображение ──► VisionBackbone (ViT/EfficientNet) ──► Патч-признаки
                        │                                    │
                   Grad-CAM++                          Cross-Attention ──► DecisionHead ──► label + confidence
                        │                                    │
Документ ──► ContextAnalyzer (RuBERT/LaBSE) ──► Токен-признаки
                                                             │
                                                     ReportGenerator (T5) ──► Отчёт
```

| Модели | Назначение |
|---|---|
| `vit_base_patch16_224` (timm) | Vision backbone |
| `DeepPavlov/rubert-base-cased` | NLP анализ текста стандарта |
| `cointegrated/rut5-base` | Генерация текстового отчёта |

Лучший чекпоинт: `models/production.pt` — val acc **81.53%** (50 эп. freeze + 20 эп. unfreeze).

---

### Pipeline B — YOLO+CLIP (нулевое обучение, Variant B)

```
Изображение ──► YOLO (yolo11n.pt)      ──► bbox-кропы (до 20)  ─┐
            └──► 3×3 сетка + full       ──► 10 зон              ─┤
                                                                  ├──► clip-ViT-B-32 ──► [N, D]
Запрос (RU) ──► _normalize_query() ──► clip-ViT-B-32-multilingual-v1 ──► [D]
                                                                  │
                                          cosine_similarity ──────┘ ──► max → label
```

| Модели | Назначение |
|---|---|
| `yolo11n.pt` (ultralytics, ~6 MB) | Детекция объектов |
| `sentence-transformers/clip-ViT-B-32` | Кодирование изображений |
| `sentence-transformers/clip-ViT-B-32-multilingual-v1` | Кодирование русских запросов |

Порог сходства: **0.25** (возвращается в ответе).
Запросы автоматически нормализуются — нормативные конструкции убираются перед CLIP:

```
"Шахматный узор должен быть справа" → "Шахматный узор справа"
"Логотип должен находиться в центре" → "Логотип в центре"
"Шахматный узор должен отсутствовать" → "Шахматный узор"
```

После CLIP-скоринга применяется **spatial boost**: если запрос содержит пространственное слово, регионы в соответствующей части изображения получают множитель ×1.15 при выборе `best_region`. Значения `confidence` и `all_similarities` остаются исходными.

| Ключевые слова | Регионы с бустом |
|---|---|
| `внизу`, `снизу`, `нижн…` | `bottom-left`, `bottom-center`, `bottom-right` |
| `сверху`, `наверху`, `верхн…`, `вверху` | `top-left`, `top-center`, `top-right` |
| `слева`, `левый/ая/ое` | `top-left`, `center-left`, `bottom-left` |
| `справа`, `правый/ая/ое` | `top-right`, `center-right`, `bottom-right` |
| `в центре`, `по центру`, `посередине`, `центральн…` | `center`, `top-center`, `bottom-center` |

---

## Быстрый старт

### Установка

```bash
git clone https://github.com/sergei-karimov/HybridQA-Net-API.git
cd HybridQA-Net

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

> **WSL2 / headless**: если возникает ошибка `libGL.so.1`, удалите `opencv-python` и оставьте только headless-версию:
> ```bash
> pip uninstall opencv-python -y
> pip install opencv-python-headless --force-reinstall
> ```

### Пример использования (Pipeline A)

```python
from src.pipeline import HybridQANet

system = HybridQANet()
result = system.analyze(
    image="product_photo.jpg",
    standard_doc="quality_standard.pdf",
    query="Проверь соответствие маркировки стандарту",
)

print(result.verdict)      # "Соответствует" / "Не соответствует"
print(result.confidence)   # float [0, 1]
print(result.report)       # Текстовый отчёт
```

### Пример использования (Pipeline B)

```python
from src.yolo_clip_pipeline import YoloCLIPAnalyzer

analyzer = YoloCLIPAnalyzer()
result = analyzer.analyze(
    image="product_photo.jpg",
    query="Шахматный узор должен быть справа в центральной части",
)

print(result.verdict)           # "Соответствует стандарту"
print(result.best_region)       # "center-right"
print(result.best_similarity)   # 0.29
print(result.normalized_query)  # "Шахматный узор справа в центральной части"
```

### Запуск API

```bash
# Локально
uvicorn api.main:app --host 0.0.0.0 --port 8000

# С автоперезагрузкой (разработка)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs

### Web UI

Браузерный интерфейс для тестирования Pipeline B без curl/Swagger:

```
http://localhost:8000/ui/
```

- Вкладка **Analyze v2** — загрузить изображение, ввести запрос, нажать «Анализировать»
- Вкладка **Conditions** — список условий `must`/`must_not`, JSON-результаты
- Canvas-оверлей поверх изображения: лучший регион — красный, остальные — жёлтый градиент, YOLO-боксы — оранжевые пунктирные прямоугольники
- Горизонтальные бары схожести по всем регионам

### Rust TUI

Терминальный интерфейс (требует установленного Rust):

```bash
cd ui/tui
cargo build --release
./target/release/hybridqa-tui --url http://localhost:8000 --username admin --password password123
```

| Клавиша | Действие |
|---|---|
| `a` | Автовход с CLI-аргументами |
| `Tab` | Переключение поля: путь → запрос → кнопка |
| `Enter` | Отправить запрос (фокус на кнопке) |
| `c` | Включить/выключить кэш |
| `q` / `Esc` | Выход |

### Docker

```bash
docker-compose up --build

# Только API (без Redis)
docker build -t hybridqa-net .
docker run -p 8000:8000 hybridqa-net
```

---

## API

### Аутентификация

```bash
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password123"}'
```

Тестовые пользователи: `admin/password123`, `user/user_password`, `readonly/readonly_pass`.

---

### Pipeline A — классические эндпоинты

#### `POST /api/v1/analyze` — анализ одного изображения

```bash
TOKEN="ваш_токен"

curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@product_photo.jpg" \
  -F "query=Проверь соответствие маркировки стандарту" \
  -F "standard_text=ГОСТ 123-2020. Требования к маркировке..."
```

#### `POST /api/v1/analyze/batch` — пакетный анализ (до 32)

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"image_b64": "<base64>", "query": "Проверка #1", "standard_text": "Стандарт..."}
    ]
  }'
```

#### `POST /api/v1/analyze/conditions` — проверка нескольких условий (до 20)

```bash
curl -X POST http://localhost:8000/api/v1/analyze/conditions \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@product_photo.jpg" \
  -F 'conditions_json=[
    {"id": "c1", "query": "Маркировка присутствует", "type": "must"},
    {"id": "c2", "query": "Повреждения упаковки", "type": "must_not"}
  ]'
```

---

### Pipeline B — YOLO+CLIP эндпоинты

#### `POST /api/v1/analyze/v2` — анализ через YOLO+CLIP

```bash
curl -X POST http://localhost:8000/api/v1/analyze/v2 \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@product_photo.jpg" \
  -F 'query=Шахматный узор должен быть справа в центральной части'
```

Ответ содержит:
- `label`, `verdict`, `confidence`, `threshold`
- `best_region`, `best_similarity` — лучшая зона
- `all_similarities` — схожесть по всем зонам (по убыванию)
- `yolo_detections` — bbox-объекты с YOLO-уверенностью и CLIP-схожестью
- `grid_regions` — 10 зон (9 из сетки + full)
- `query` — оригинальный запрос
- `normalized_query` — запрос после нормализации (что реально отправлено в CLIP)

#### `POST /api/v1/analyze/v2/conditions` — YOLO+CLIP, несколько условий

```bash
curl -X POST http://localhost:8000/api/v1/analyze/v2/conditions \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@product_photo.jpg" \
  -F 'conditions_json=[
    {"id": "c1", "query": "Шахматный узор в правой части", "type": "must"},
    {"id": "c2", "query": "Шахматный узор должен отсутствовать", "type": "must_not"}
  ]'
```

---

## Fine-tuning (Pipeline A)

### Структура датасета

```
data/
├── train/
│   ├── images/
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

Формат `annotations.json`:

```json
[
  {
    "image": "images/product_001.jpg",
    "query": "Проверь соответствие маркировки",
    "document": "ГОСТ 12345-2020 ...",
    "label": 1
  }
]
```

`label`: `1` = соответствует, `0` = не соответствует.

### Генерация синтетического датасета

```bash
python tools/generate_dataset.py --count 300
```

### Запуск обучения

```python
from src.pipeline import HybridQANet
from training.dataset import QADataset
from training.trainer import Trainer

system = HybridQANet()
trainer = Trainer(system)

train_ds = QADataset("data", split="train", augment=True)
val_ds = QADataset("data", split="val")

history = trainer.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    freeze_backbone=True,
)
```

---

## Конфигурация

Файл `configs/config.yaml` — основной конфигурационный файл.

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model.vision.backbone` | `vit_base_patch16_224` | Vision backbone |
| `model.nlp.model_name` | `DeepPavlov/rubert-base-cased` | NLP модель |
| `model.report.model_name` | `cointegrated/rut5-base` | T5 для отчётов |
| `cache.backend` | `memory` | `memory` или `redis` |
| `api.rate_limit_per_minute` | `100` | Rate limit |

---

## Тесты

```bash
python -m pytest tests/ -v

# Только быстрые тесты (без загрузки моделей)
python -m pytest tests/test_pipeline.py -v
```

---

## Структура проекта

```
HybridQA-Net/
├── src/
│   ├── vision/
│   │   ├── backbone.py          # ViT / EfficientNet бэкбон
│   │   ├── gradcam.py           # Grad-CAM++ визуализация
│   │   ├── preprocessor.py      # Предобработка изображений
│   │   ├── detector.py          # YOLODetector + 3×3 grid (Variant B)
│   │   └── clip_matcher.py      # CLIPMatcher: image+text encoders (Variant B)
│   ├── nlp/
│   │   ├── context_analyzer.py  # RuBERT / LaBSE анализатор
│   │   └── document_parser.py   # Парсер PDF/TXT
│   ├── fusion/
│   │   ├── cross_attention.py   # Cross-Attention слияние
│   │   └── decision_head.py     # Классификатор + confidence
│   ├── report/
│   │   └── generator.py         # T5 генератор отчётов
│   ├── cache.py                 # Кэширование (memory / redis)
│   ├── pipeline.py              # Pipeline A: HybridQANet
│   └── yolo_clip_pipeline.py    # Pipeline B: YoloCLIPAnalyzer
├── api/
│   ├── main.py                  # FastAPI приложение (оба pipeline)
│   ├── routes.py                # Маршруты v1 и v2
│   ├── middleware.py            # Auth + Rate limiting
│   └── schemas.py               # Pydantic схемы
├── training/
│   ├── dataset.py               # PyTorch Dataset
│   └── trainer.py               # Цикл обучения
├── tools/
│   └── generate_dataset.py      # Генерация синтетического датасета
├── utils/
│   ├── logger.py                # Логирование
│   └── helpers.py               # Вспомогательные функции
├── ui/
│   ├── web/
│   │   └── index.html           # Web UI (тёмная тема, canvas overlay, бары)
│   └── tui/
│       ├── Cargo.toml           # Rust TUI зависимости
│       └── src/main.rs          # ratatui: grid table + bar chart
├── tests/                       # Тесты
├── configs/config.yaml          # Конфигурация
├── example_usage.py             # Пример использования
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Лицензия

MIT
