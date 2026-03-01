"""
Генератор синтетического датасета для HybridQA-Net.

Создаёт изображения этикеток с двумя независимыми декоративными зонами
(A — правый верхний, B — правый центральный) и генерирует по 12 аннотаций
на каждое изображение, охватывающих все целевые условия проверки.

Использование:
    python tools/generate_dataset.py --output data --count 300 --val-ratio 0.2
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ================================================================= Размеры
LABEL_W, LABEL_H = 600, 400

# Зоны декоративных элементов (правая часть изображения)
ZONE_A = {"x0": 440, "y0": 100, "x1": 590, "y1": 220}  # верхняя правая
ZONE_B = {"x0": 440, "y0": 225, "x1": 590, "y1": 345}  # центральная правая

GOSTS = ["ГОСТ 12345-2020", "ГОСТ 54321-2019", "ГОСТ 99999-2021"]
PRODUCTS = [
    "Шоколад молочный", "Печенье сдобное", "Йогурт натуральный",
    "Сыр твёрдый", "Масло сливочное", "Кефир 2.5%", "Творог обезжиренный",
]
WEIGHTS = ["250 г", "500 г", "1 кг", "330 мл", "1 л", "750 г"]
SHELF_LIVES = ["6 месяцев", "12 месяцев", "18 месяцев", "24 месяца"]

REQUIRED_MARKS = {
    "temperature": "Хранить при t < +25°C",
    "humidity":    "Влажность не более 75%",
    "light":       "Беречь от прямых солнечных лучей",
    "children":    "Хранить в недоступном для детей месте",
}

COLORS = {
    "header_bg": [(30, 80, 160), (160, 30, 30), (30, 130, 60), (100, 50, 150)],
    "body_bg":   [(245, 245, 220), (255, 255, 240), (240, 248, 255), (255, 250, 240)],
    "text":      [(20, 20, 20), (40, 40, 40)],
}

# Типы зональных элементов и их допустимые размеры
ZONE_TYPES = [None, "chess", "oval", "stripes"]
CHESS_SIZES = ["3x3", "4x4", "6x6"]


# ================================================================= Шрифты
def get_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ================================================================= Рисование зон
def _draw_chess(draw: ImageDraw.Draw, zone: dict, size: str) -> None:
    """Нарисовать шахматный узор в зоне."""
    n = int(size.split("x")[0])
    zw = zone["x1"] - zone["x0"]
    zh = zone["y1"] - zone["y0"]
    cell_w = zw // n
    cell_h = zh // n
    for row in range(n):
        for col in range(n):
            color = (20, 20, 20) if (row + col) % 2 == 0 else (240, 240, 240)
            x0 = zone["x0"] + col * cell_w
            y0 = zone["y0"] + row * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=color)


def _draw_oval(draw: ImageDraw.Draw, zone: dict, rng: random.Random) -> None:
    """Нарисовать овал в зоне."""
    c = rng.randint(80, 180)
    draw.ellipse(
        [zone["x0"] + 5, zone["y0"] + 5, zone["x1"] - 5, zone["y1"] - 5],
        fill=(c, c, c),
    )


def _draw_stripes(draw: ImageDraw.Draw, zone: dict, rng: random.Random) -> None:
    """Нарисовать горизонтальные полосы в зоне."""
    zh = zone["y1"] - zone["y0"]
    n_stripes = 6
    stripe_h = zh // n_stripes
    for i in range(n_stripes):
        c = rng.randint(50, 200)
        y0 = zone["y0"] + i * stripe_h
        draw.rectangle([zone["x0"], y0, zone["x1"], y0 + stripe_h - 2], fill=(c, c, c))


def _draw_zone(draw: ImageDraw.Draw, zone: dict, zone_info: dict, rng: random.Random) -> None:
    """Нарисовать зональный элемент по описанию."""
    ztype = zone_info["type"]
    if ztype == "chess":
        _draw_chess(draw, zone, zone_info["size"])
    elif ztype == "oval":
        _draw_oval(draw, zone, rng)
    elif ztype == "stripes":
        _draw_stripes(draw, zone, rng)
    # None — ничего не рисуем


# ================================================================= Генерация manifest
def _make_manifest(rng: random.Random) -> dict:
    """Сформировать случайный манифест для изображения."""
    def random_zone():
        ztype = rng.choice(ZONE_TYPES)
        size = rng.choice(CHESS_SIZES) if ztype == "chess" else None
        return {"type": ztype, "size": size}

    return {
        "zone_a": random_zone(),
        "zone_b": random_zone(),
        "marks": {k: rng.random() < 0.5 for k in REQUIRED_MARKS},
    }


# ================================================================= Генерация изображения
def generate_label(manifest: dict, seed: int = 0) -> Image.Image:
    """Сгенерировать этикетку по манифесту."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    img = Image.new("RGB", (LABEL_W, LABEL_H), (230, 230, 230))
    draw = ImageDraw.Draw(img)

    border_color = rng.choice(COLORS["header_bg"])
    draw.rectangle([2, 2, LABEL_W - 3, LABEL_H - 3], outline=border_color, width=3)

    header_color = rng.choice(COLORS["header_bg"])
    draw.rectangle([10, 10, LABEL_W - 10, 90], fill=header_color)

    product = rng.choice(PRODUCTS)
    gost = rng.choice(GOSTS)

    f_big   = get_font(22)
    f_small = get_font(13)

    draw.text((20, 18), product, font=f_big,   fill=(255, 255, 255))
    draw.text((20, 55), gost,   font=f_small,  fill=(200, 220, 255))

    body_color = rng.choice(COLORS["body_bg"])
    draw.rectangle([10, 95, LABEL_W - 160, LABEL_H - 10], fill=body_color)

    text_x, text_y = 20, 105
    text_color = rng.choice(COLORS["text"])

    draw.text((text_x, text_y),      "Состав: натуральные ингредиенты",
              font=f_small, fill=text_color)
    draw.text((text_x, text_y + 25), f"Масса: {rng.choice(WEIGHTS)}",
              font=f_small, fill=text_color)
    draw.text((text_x, text_y + 50), f"Срок годности: {rng.choice(SHELF_LIVES)}",
              font=f_small, fill=text_color)

    # Метки хранения
    mark_y = text_y + 80
    for key, text in REQUIRED_MARKS.items():
        if manifest["marks"][key]:
            draw.text((text_x, mark_y), text, font=f_small, fill=(180, 0, 0))
        mark_y += 22

    # Штрихкод
    bc_x, bc_y = 20, LABEL_H - 65
    for i in range(50):
        width = rng.randint(1, 3)
        x = bc_x + i * 5
        draw.rectangle([x, bc_y, x + width, bc_y + 40], fill=(10, 10, 10))
    draw.text(
        (bc_x, bc_y + 42),
        f"4607 {rng.randint(100,999)} {rng.randint(100,999)} {rng.randint(100,999)}",
        font=f_small, fill=text_color,
    )

    # Фон правой части
    draw.rectangle([LABEL_W - 160, 95, LABEL_W - 10, LABEL_H - 10],
                   fill=rng.choice(COLORS["body_bg"]))

    # Декоративные зоны
    _draw_zone(draw, ZONE_A, manifest["zone_a"], rng)
    _draw_zone(draw, ZONE_B, manifest["zone_b"], rng)

    # Аугментация
    img = _augment(img, rng, np_rng)
    return img


# ================================================================= Аугментация
def _augment(img: Image.Image, rng: random.Random, np_rng) -> Image.Image:
    angle = rng.uniform(-3, 3)
    img = img.rotate(angle, fillcolor=(200, 200, 200), expand=False)

    arr = np.array(img).astype(np.float32)
    brightness = np_rng.uniform(0.85, 1.15)
    noise = np_rng.normal(0, rng.uniform(0, 8), arr.shape)
    arr = (arr * brightness + noise).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    return img


# ================================================================= Аннотации
DOCUMENT = (
    f"Требования стандарта: {', '.join(GOSTS)}. "
    "Обязательная маркировка условий хранения. "
    "Отсутствие любой обязательной метки является нарушением."
)

# (condition_key, query_text, label_fn(manifest) -> bool)
CONDITION_DEFS = [
    (
        "chess_top",
        "Шахматный узор в правом верхнем углу",
        lambda m: m["zone_a"]["type"] == "chess",
    ),
    (
        "chess_size_3x3",
        "Шахматный узор 3×3 клетки",
        lambda m: m["zone_a"]["type"] == "chess" and m["zone_a"]["size"] == "3x3",
    ),
    (
        "chess_size_4x4",
        "Шахматный узор 4×4 клетки",
        lambda m: m["zone_a"]["type"] == "chess" and m["zone_a"]["size"] == "4x4",
    ),
    (
        "chess_size_6x6",
        "Шахматный узор 6×6 клеток",
        lambda m: m["zone_a"]["type"] == "chess" and m["zone_a"]["size"] == "6x6",
    ),
    (
        "chess_absent",
        "Шахматный узор должен отсутствовать",
        lambda m: m["zone_a"]["type"] != "chess" and m["zone_b"]["type"] != "chess",
    ),
    (
        "oval_present",
        "Овал должен присутствовать справа",
        lambda m: m["zone_a"]["type"] == "oval" or m["zone_b"]["type"] == "oval",
    ),
    (
        "oval_below_chess",
        "Под шахматным узором должен быть овал",
        lambda m: m["zone_a"]["type"] == "chess" and m["zone_b"]["type"] == "oval",
    ),
    (
        "nothing_decorative",
        "Не должно быть графических элементов",
        lambda m: m["zone_a"]["type"] is None and m["zone_b"]["type"] is None,
    ),
    (
        "temperature",
        REQUIRED_MARKS["temperature"],
        lambda m: m["marks"]["temperature"],
    ),
    (
        "humidity",
        REQUIRED_MARKS["humidity"],
        lambda m: m["marks"]["humidity"],
    ),
    (
        "light",
        REQUIRED_MARKS["light"],
        lambda m: m["marks"]["light"],
    ),
    (
        "children",
        REQUIRED_MARKS["children"],
        lambda m: m["marks"]["children"],
    ),
]


def build_annotations(image_path: str, manifest: dict) -> list[dict]:
    """Сгенерировать 12 аннотаций для одного изображения."""
    entries = []
    for cond_key, query, label_fn in CONDITION_DEFS:
        entries.append({
            "image":         image_path,
            "query":         query,
            "document":      DOCUMENT,
            "label":         1 if label_fn(manifest) else 0,
            "condition_key": cond_key,
        })
    return entries


# ================================================================= Генерация датасета
def generate_dataset(output_dir: Path, count: int, val_ratio: float) -> None:
    n_val = int(count * val_ratio)
    n_train = count - n_val

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = output_dir / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        annotations = []
        for i in range(n):
            seed = hash((split, i)) % (2**31)
            rng = random.Random(seed)
            manifest = _make_manifest(rng)

            img = generate_label(manifest, seed=seed)
            filename = f"sample_{i:04d}.jpg"
            img.save(img_dir / filename, quality=92)

            entries = build_annotations(f"images/{filename}", manifest)
            annotations.extend(entries)

        ann_path = output_dir / split / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)

        pos = sum(a["label"] for a in annotations)
        total = len(annotations)
        print(
            f"[{split}] {n} изображений × 12 условий = {total} аннотаций | "
            f"✓ {pos} ({pos/total:.1%}) | ✗ {total - pos} ({(total-pos)/total:.1%})"
        )
        print(f"  → {ann_path}")


# ================================================================= CLI
def main():
    parser = argparse.ArgumentParser(description="Генератор датасета HybridQA-Net")
    parser.add_argument("--output",    default="data",        help="Выходная директория")
    parser.add_argument("--count",     type=int, default=300, help="Число изображений")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Доля val")
    args = parser.parse_args()

    output = Path(args.output)
    print(f"Генерация датасета: {args.count} изображений → {output}/")
    print(f"Условий на изображение: {len(CONDITION_DEFS)}")
    print()

    generate_dataset(output, args.count, args.val_ratio)
    print("\nГотово!")


if __name__ == "__main__":
    main()
