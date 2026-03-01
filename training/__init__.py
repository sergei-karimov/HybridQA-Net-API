"""Модуль обучения HybridQA-Net."""

from .dataset import QADataset, collate_fn
from .trainer import Trainer

__all__ = ["QADataset", "collate_fn", "Trainer"]
