"""Настройка логирования для HybridQA-Net."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import yaml


def setup_logger(
    name: str,
    config_path: str = "configs/config.yaml",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Создать и настроить логгер.

    Args:
        name: Имя логгера.
        config_path: Путь к файлу конфигурации.
        level: Уровень логирования (переопределяет конфиг).

    Returns:
        Настроенный логгер.
    """
    logger = logging.getLogger(name)

    if logger.handlers:  # Не добавлять дублирующие обработчики
        return logger

    # Загрузить конфигурацию
    log_level = level or "INFO"
    log_file = "./logs/app.log"
    max_bytes = 10_485_760
    backup_count = 5

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        log_cfg = config.get("logging", {})
        log_level = level or log_cfg.get("level", "INFO")
        log_file = log_cfg.get("file", "./logs/app.log")
        max_bytes = log_cfg.get("max_bytes", 10_485_760)
        backup_count = log_cfg.get("backup_count", 5)
    except (FileNotFoundError, KeyError):
        pass

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    # Отключаем передачу сообщений родительским логгерам во избежание дублирования
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Вывод в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Вывод в файл с ротацией
    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        logger.warning(f"Не удалось создать файловый лог: {e}")

    return logger


# Корневой логгер приложения
logger = setup_logger("hybridqa")
