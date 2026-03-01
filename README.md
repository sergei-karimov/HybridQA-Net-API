# HybridQA-Net

Гибридная мультимодальная система контроля качества на базе Vision Transformer и RuBERT.

## Архитектура

```
Изображение ──► VisionBackbone (ViT/EfficientNet) ──► Патч-признаки
                        │                                    │
                   Grad-CAM++                          Cross-Attention ──► DecisionHead ──► label + confidence
                        │                                    │
Документ ──► ContextAnalyzer (RuBERT/LaBSE) ──► Токен-признаки
                                                             │
                                                     ReportGenerator (T5) ──► Отчёт
```

## Быстрый старт

### Установка

```bash
# Клонируем репозиторий
git clone https://github.com/your-org/HybridQA-Net.git
cd HybridQA-Net

# Создаём виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Устанавливаем зависимости
pip install -r requirements.txt
```

### Пример использования

```python
from src.pipeline import HybridQANet

system = HybridQANet()
result = system.analyze(
    image="product_photo.jpg",
    standard_doc="quality_standard.pdf",
    query="Проверь соответствие маркировки стандарту",
)

print(result.report)          # Текстовый отчёт
print(result.attention_map)   # Numpy массив [H, W]
print(result.confidence)      # float [0, 1]
print(result.verdict)         # "Соответствует" / "Не соответствует"
```

Запустить полный пример:

```bash
python example_usage.py
```

### Запуск API

```bash
# Локально
uvicorn api.main:app --host 0.0.0.0 --port 8000

# С автоперезагрузкой (для разработки)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI доступен по адресу: http://localhost:8000/docs

### Docker

```bash
# Собрать и запустить
docker-compose up --build

# Только API (без Redis)
docker build -t hybridqa-net .
docker run -p 8000:8000 hybridqa-net
```

## API

### Аутентификация

```bash
# Получить токен
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password123"}'
```

### Анализ изображения

```bash
TOKEN="ваш_токен"

curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@product_photo.jpg" \
  -F "query=Проверь соответствие маркировки стандарту" \
  -F "standard_text=ГОСТ 123-2020. Требования к маркировке..."
```

### Пакетный анализ

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "image_b64": "<base64_image>",
        "query": "Проверка #1",
        "standard_text": "Стандарт..."
      }
    ]
  }'
```

## Fine-tuning

### Структура датасета

```
data/
├── train/
│   ├── images/
│   │   ├── product_001.jpg
│   │   └── ...
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
    freeze_backbone=True,   # Обучать только голову
)
```

## Конфигурация

Файл `configs/config.yaml` — основной конфигурационный файл.

Ключевые параметры:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model.vision.backbone` | `vit_base_patch16_224` | Vision backbone (ViT или EfficientNet) |
| `model.nlp.model_name` | `DeepPavlov/rubert-base-cased` | NLP модель |
| `model.report.model_name` | `cointegrated/rut5-base` | T5 для генерации отчётов |
| `cache.backend` | `memory` | Бэкенд кэша (`memory` или `redis`) |
| `api.rate_limit_per_minute` | `100` | Rate limit |

## Тесты

```bash
# Запуск всех тестов
python -m pytest tests/ -v

# Только быстрые тесты (без загрузки моделей)
python -m pytest tests/test_pipeline.py -v
```

## Структура проекта

```
HybridQA-Net/
├── src/
│   ├── vision/
│   │   ├── backbone.py        # ViT / EfficientNet бэкбон
│   │   ├── gradcam.py         # Grad-CAM++ визуализация
│   │   └── preprocessor.py    # Предобработка изображений
│   ├── nlp/
│   │   ├── context_analyzer.py # RuBERT / LaBSE анализатор
│   │   └── document_parser.py  # Парсер PDF/TXT
│   ├── fusion/
│   │   ├── cross_attention.py  # Cross-Attention слияние
│   │   └── decision_head.py    # Классификатор + confidence
│   ├── report/
│   │   └── generator.py        # T5 генератор отчётов
│   ├── cache.py                # Кэширование (memory / redis)
│   └── pipeline.py             # Главный pipeline
├── api/
│   ├── main.py                 # FastAPI приложение
│   ├── routes.py               # API маршруты
│   ├── middleware.py           # Auth + Rate limiting
│   └── schemas.py              # Pydantic схемы
├── training/
│   ├── dataset.py              # PyTorch Dataset
│   └── trainer.py              # Цикл обучения
├── utils/
│   ├── logger.py               # Логирование
│   └── helpers.py              # Вспомогательные функции
├── tests/                      # Тесты
├── configs/config.yaml         # Конфигурация
├── example_usage.py            # Пример использования
├── Dockerfile                  # Docker образ
├── docker-compose.yml          # Docker Compose
└── requirements.txt            # Зависимости
```

## Лицензия

MIT
