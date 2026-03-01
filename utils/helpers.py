"""Вспомогательные утилиты."""

import hashlib
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Загрузить YAML-конфигурацию."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> torch.device:
    """
    Автоматически выбрать устройство (CUDA / MPS / CPU).

    Args:
        config: Словарь конфигурации.

    Returns:
        torch.device для вычислений.
    """
    if not config.get("device", {}).get("auto_select", True):
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_hash(data: bytes) -> str:
    """SHA-256 хэш бинарных данных (для кэш-ключей)."""
    return hashlib.sha256(data).hexdigest()


def compute_text_hash(text: str) -> str:
    """SHA-256 хэш строки."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def timer(func: Callable) -> Callable:
    """Декоратор для замера времени выполнения функции."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        # Логируем через стандартный print, чтобы не создавать циклических импортов
        print(f"[timer] {func.__qualname__} выполнена за {elapsed:.3f}с")
        return result

    return wrapper


def ensure_dir(path: str | Path) -> Path:
    """Создать директорию, если не существует."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
