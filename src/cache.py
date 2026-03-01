"""
Модуль кэширования результатов анализа.

Поддерживает два бэкенда:
- ``memory`` — in-memory LRU-кэш (нет зависимостей, подходит для одного процесса)
- ``redis`` — распределённый кэш через Redis (для multi-worker/multi-instance)
"""

from __future__ import annotations

import json
import pickle
import time
from collections import OrderedDict
from typing import Any, Optional


class MemoryCache:
    """
    Thread-safe LRU кэш в памяти.

    Args:
        max_items: Максимальное число элементов.
        ttl_seconds: Время жизни записи в секундах.
    """

    def __init__(self, max_items: int = 512, ttl_seconds: int = 3600):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self.max_items = max_items
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша. None если не найдено или устарело."""
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        # Перемещаем в конец (LRU)
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        """Сохранить значение в кэше."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        if len(self._store) > self.max_items:
            self._store.popitem(last=False)  # Удалить самый старый

    def delete(self, key: str) -> None:
        """Удалить запись."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Очистить весь кэш."""
        self._store.clear()

    def stats(self) -> dict:
        """Статистика кэша."""
        return {"size": len(self._store), "max_items": self.max_items}


class RedisCache:
    """
    Кэш на базе Redis.

    Args:
        redis_url: URL подключения к Redis.
        ttl_seconds: Время жизни ключей.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 3600):
        try:
            import redis as redis_lib
            self._client = redis_lib.from_url(redis_url, decode_responses=False)
        except ImportError:
            raise ImportError("Установите redis: pip install redis")
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из Redis."""
        data = self._client.get(key)
        if data is None:
            return None
        return pickle.loads(data)

    def set(self, key: str, value: Any) -> None:
        """Сохранить значение в Redis."""
        self._client.setex(key, self.ttl, pickle.dumps(value))

    def delete(self, key: str) -> None:
        """Удалить ключ."""
        self._client.delete(key)

    def clear(self) -> None:
        """Флаш всей базы (осторожно в продакшне!)."""
        self._client.flushdb()

    def stats(self) -> dict:
        """Статистика Redis."""
        info = self._client.info("memory")
        return {
            "used_memory_human": info.get("used_memory_human"),
            "connected_clients": self._client.info().get("connected_clients"),
        }


class CacheManager:
    """
    Единый интерфейс для работы с кэшем.

    Автоматически выбирает бэкенд по конфигурации.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Секция ``cache`` из config.yaml.
        """
        self.enabled = config.get("enabled", True)
        backend = config.get("backend", "memory")

        if not self.enabled:
            self._cache = None
            return

        if backend == "redis":
            self._cache = RedisCache(
                redis_url=config.get("redis_url", "redis://localhost:6379/0"),
                ttl_seconds=config.get("ttl_seconds", 3600),
            )
        else:
            self._cache = MemoryCache(
                max_items=config.get("max_memory_items", 512),
                ttl_seconds=config.get("ttl_seconds", 3600),
            )

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша."""
        if not self.enabled or self._cache is None:
            return None
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Записать значение в кэш."""
        if not self.enabled or self._cache is None:
            return
        self._cache.set(key, value)

    def delete(self, key: str) -> None:
        """Удалить запись из кэша."""
        if self._cache:
            self._cache.delete(key)

    def clear(self) -> None:
        """Очистить весь кэш."""
        if self._cache:
            self._cache.clear()

    def stats(self) -> dict:
        """Получить статистику кэша."""
        if not self.enabled or self._cache is None:
            return {"enabled": False}
        return {"enabled": True, **self._cache.stats()}
