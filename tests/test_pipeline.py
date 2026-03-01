"""
Интеграционные тесты pipeline HybridQA-Net.

Тесты используют stub-компоненты чтобы не загружать реальные модели.
"""

import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image


def _make_jpeg_bytes(size: tuple[int, int] = (224, 224)) -> bytes:
    """Создать минимальный JPEG в байтах."""
    img = Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestCacheManager(unittest.TestCase):
    def test_memory_cache_set_get(self):
        from src.cache import MemoryCache
        cache = MemoryCache(max_items=10, ttl_seconds=60)
        cache.set("key1", {"value": 42})
        result = cache.get("key1")
        self.assertEqual(result, {"value": 42})

    def test_memory_cache_expiry(self):
        import time
        from src.cache import MemoryCache
        cache = MemoryCache(max_items=10, ttl_seconds=1)
        cache.set("key1", "data")
        time.sleep(1.1)
        self.assertIsNone(cache.get("key1"))

    def test_memory_cache_lru_eviction(self):
        from src.cache import MemoryCache
        cache = MemoryCache(max_items=3, ttl_seconds=60)
        for i in range(4):
            cache.set(f"key{i}", i)
        # Первый ключ должен быть вытеснен
        self.assertIsNone(cache.get("key0"))

    def test_cache_manager_disabled(self):
        from src.cache import CacheManager
        manager = CacheManager({"enabled": False})
        manager.set("k", "v")
        self.assertIsNone(manager.get("k"))


class TestFusionModules(unittest.TestCase):
    def test_cross_attention_output_shape(self):
        from src.fusion.cross_attention import CrossAttentionFusion
        fusion = CrossAttentionFusion(embed_dim=64, num_heads=4, num_layers=2)
        visual = torch.randn(2, 196, 64)
        text = torch.randn(2, 32, 64)
        out = fusion(visual, text)
        self.assertEqual(out.fused_embedding.shape, (2, 64))

    def test_decision_head_output(self):
        from src.fusion.decision_head import DecisionHead
        head = DecisionHead(input_dim=64, hidden_dim=32, num_classes=2)
        emb = torch.randn(1, 64)
        out = head(emb)
        self.assertIn(out.label, [0, 1])
        self.assertGreaterEqual(out.confidence, 0.0)
        self.assertLessEqual(out.confidence, 1.0)

    def test_cross_attention_with_global_vectors(self):
        from src.fusion.cross_attention import CrossAttentionFusion
        fusion = CrossAttentionFusion(embed_dim=64, num_heads=4, num_layers=1)
        # Проверяем что работает с 2D тензорами [B, D]
        visual = torch.randn(2, 64)
        text = torch.randn(2, 64)
        out = fusion(visual, text)
        self.assertEqual(out.fused_embedding.shape, (2, 64))


class TestDocumentParser(unittest.TestCase):
    def test_parse_plain_text_from_string(self):
        from src.nlp.document_parser import DocumentParser
        parser = DocumentParser()
        result = parser.parse(
            "Стандарт\nТребования".encode("utf-8"),
            file_type="txt",
        )
        self.assertIn("Стандарт", result)

    def test_parse_chunks(self):
        from src.nlp.document_parser import DocumentParser
        parser = DocumentParser()
        # Для chunks нам нужен файл — тест с моком
        long_text = "AB" * 300
        chunks = parser._clean_text(long_text)
        self.assertGreater(len(chunks), 0)


if __name__ == "__main__":
    unittest.main()
