"""
Dataset класс для fine-tuning HybridQA-Net.

Ожидаемая структура данных:
    data/
    ├── train/
    │   ├── images/
    │   │   ├── sample_001.jpg
    │   │   └── ...
    │   └── annotations.json
    └── val/
        ├── images/
        └── annotations.json

Формат annotations.json:
    [
        {
            "image": "images/sample_001.jpg",
            "query": "Проверь соответствие маркировки",
            "document": "ГОСТ 12345-2020 ...",
            "label": 1
        },
        ...
    ]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from src.vision.preprocessor import ImagePreprocessor


class QADataset(Dataset):
    """
    Dataset для задачи контроля качества.

    Args:
        data_root: Корневая директория с данными.
        split: "train" или "val".
        augment: Применять аугментацию (только для train).
        max_doc_length: Максимальная длина текста документа.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        augment: bool = False,
        max_doc_length: int = 512,
        image_size: int = 224,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_doc_length = max_doc_length

        # Загрузка аннотаций
        ann_path = self.data_root / split / "annotations.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Файл аннотаций не найден: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
            augment=augment and split == "train",
        )

        # Статистика классов для взвешивания
        labels = [item["label"] for item in self.annotations]
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos
        total = len(labels)

        # Веса классов для несбалансированных данных
        self.class_weights = torch.tensor(
            [total / (2 * num_neg + 1e-8), total / (2 * num_pos + 1e-8)],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        """
        Вернуть один элемент датасета.

        Returns:
            Словарь с ключами:
            - pixel_values: [3, H, W]
            - query: str
            - document: str (обрезанный до max_doc_length)
            - label: int (0 или 1)
        """
        item = self.annotations[idx]

        # Загрузка изображения
        image_path = self.data_root / self.split / item["image"]
        pixel_values = self.preprocessor.preprocess(image_path).squeeze(0)

        # Текст
        query = item.get("query", "Проверь соответствие")
        document = item.get("document", "")[:self.max_doc_length]
        label = int(item["label"])

        return {
            "pixel_values": pixel_values,
            "query": query,
            "document": document,
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Функция склейки батча.

    Изображения стэкаются в тензор, строки остаются списками.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    queries = [item["query"] for item in batch]
    documents = [item["document"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "queries": queries,
        "documents": documents,
        "labels": labels,
    }
