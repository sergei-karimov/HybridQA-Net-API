"""
Пример использования HybridQA-Net.

Запуск:
    python example_usage.py

Требования:
    - Установлены зависимости из requirements.txt
    - Доступны модели (будут скачаны автоматически с HuggingFace)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import io
import numpy as np
from PIL import Image


# ============================================================= Создание тестового изображения
def create_test_image(path: str = "/tmp/test_product.jpg") -> str:
    """Создать простое тестовое изображение продукта."""
    # Имитируем изображение продукта с цветными блоками
    img = Image.new("RGB", (400, 400), color=(200, 200, 200))
    arr = np.array(img)

    # Добавляем "маркировку" — белый прямоугольник
    arr[50:150, 50:350] = [255, 255, 255]
    # Добавляем "дефект" — тёмное пятно
    arr[200:250, 150:200] = [30, 30, 30]

    Image.fromarray(arr).save(path, quality=90)
    return path


# ============================================================= Основной пример
def main():
    print("=" * 60)
    print("HybridQA-Net — Пример использования")
    print("=" * 60)

    # ---- 1. Инициализация системы
    print("\n[1/4] Инициализация HybridQA-Net...")
    from src.pipeline import HybridQANet

    system = HybridQANet(config_path="configs/config.yaml")
    print("    ✓ Система инициализирована")

    # ---- 2. Подготовка тестовых данных
    print("\n[2/4] Подготовка тестовых данных...")
    image_path = create_test_image()
    print(f"    ✓ Тестовое изображение создано: {image_path}")

    standard_text = """
    СТАНДАРТ КАЧЕСТВА СК-2024-001

    1. ТРЕБОВАНИЯ К МАРКИРОВКЕ
    1.1. Маркировка должна содержать наименование продукта.
    1.2. Текст должен быть чётким и читаемым.
    1.3. Фоновый цвет: белый (RGB 255, 255, 255).

    2. ТРЕБОВАНИЯ К УПАКОВКЕ
    2.1. Отсутствие тёмных пятен и загрязнений.
    2.2. Поверхность однородного цвета.
    """
    print("    ✓ Текст стандарта подготовлен")

    # ---- 3. Анализ одного изображения
    print("\n[3/4] Анализ изображения...")
    result = system.analyze(
        image=image_path,
        standard_doc=standard_text,
        query="Проверь соответствие маркировки и наличие дефектов поверхности",
    )

    print("\n" + "─" * 40)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("─" * 40)
    print(f"Вердикт:         {result.verdict}")
    print(f"Уверенность:     {result.confidence:.1%}")
    print(f"Оценка дефекта:  {result.defect_score:.3f}")
    print(f"Время анализа:   {result.processing_time_s:.2f}с")
    print(f"Из кэша:         {result.cached}")
    print()
    print("ОТЧЁТ:")
    print(result.report)

    if result.defects:
        print("ДЕФЕКТЫ:")
        for d in result.defects:
            print(f"  • {d}")

    print("\nРЕКОМЕНДАЦИИ:")
    for r in result.recommendations:
        print(f"  → {r}")

    # Сохранение карты внимания
    if result.overlay_image is not None:
        attention_path = "/tmp/attention_map.png"
        Image.fromarray(result.overlay_image).save(attention_path)
        print(f"\n✓ Карта внимания сохранена: {attention_path}")

    # ---- 4. Повторный запрос (из кэша)
    print("\n[4/4] Повторный запрос (должен вернуться из кэша)...")
    result_cached = system.analyze(
        image=image_path,
        standard_doc=standard_text,
        query="Проверь соответствие маркировки и наличие дефектов поверхности",
    )
    print(f"    Из кэша: {result_cached.cached} ✓")

    # ---- 5. Пакетная обработка
    print("\n[5] Демо пакетной обработки...")
    batch_items = [
        {
            "image": image_path,
            "standard_doc": standard_text,
            "query": f"Проверка #{i+1}",
        }
        for i in range(3)
    ]
    batch_results = system.analyze_batch(batch_items)
    print(f"    Обработано: {len(batch_results)} элементов")
    for i, r in enumerate(batch_results):
        print(f"    [{i+1}] {r.verdict} ({r.confidence:.1%})")

    print("\n" + "=" * 60)
    print("Пример завершён успешно!")


# ============================================================= Fine-tuning пример
def fine_tuning_example():
    """Демонстрация fine-tuning (требует датасет в data/)."""
    print("\n=== Fine-tuning Example ===")

    from src.pipeline import HybridQANet
    from training.dataset import QADataset
    from training.trainer import Trainer

    system = HybridQANet()
    trainer = Trainer(system)

    # Создаём примерную структуру данных
    import json
    from pathlib import Path

    Path("data/train/images").mkdir(parents=True, exist_ok=True)
    Path("data/val/images").mkdir(parents=True, exist_ok=True)

    # Фиктивные аннотации (в реальности нужны реальные данные)
    annotations = [
        {
            "image": "images/sample.jpg",
            "query": "Проверь соответствие маркировки",
            "document": "ГОСТ 123. Требования: чёткая маркировка.",
            "label": 1,
        }
    ]

    # Создаём test image
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    img.save("data/train/images/sample.jpg")
    img.save("data/val/images/sample.jpg")

    with open("data/train/annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False)
    with open("data/val/annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False)

    train_ds = QADataset("data", split="train", augment=True)
    val_ds = QADataset("data", split="val")

    print(f"Train размер: {len(train_ds)}, Val размер: {len(val_ds)}")

    # trainer.train(train_ds, val_ds)   # Раскомментировать для реального обучения
    print("Fine-tuning готов к запуску (раскомментируйте trainer.train)")


if __name__ == "__main__":
    main()
    # fine_tuning_example()   # Раскомментировать для демо fine-tuning
