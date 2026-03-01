"""Тесты NLP модуля."""

import unittest

import torch

from src.nlp.context_analyzer import ContextAnalyzer
from src.nlp.document_parser import DocumentParser


class TestDocumentParser(unittest.TestCase):
    def setUp(self):
        self.parser = DocumentParser()

    def test_parse_txt_bytes(self):
        text = "Стандарт качества ГОСТ 123.\nПункт 1: требования к маркировке."
        result = self.parser.parse(text.encode("utf-8"), file_type="txt")
        self.assertIn("ГОСТ", result)

    def test_clean_text(self):
        raw = "  Hello   \n\n\n\nWorld  "
        cleaned = DocumentParser._clean_text(raw)
        self.assertEqual(cleaned, "Hello\n\nWorld")

    def test_parse_to_chunks(self):
        text = "A" * 2000
        with self.assertRaises(FileNotFoundError):
            self.parser.parse_to_chunks("nonexistent.txt", chunk_size=100, overlap=10)

    def test_detect_pdf_signature(self):
        result = DocumentParser._detect_type_from_bytes(b"%PDFSomePDFContent")
        self.assertEqual(result, "pdf")

    def test_detect_txt_fallback(self):
        result = DocumentParser._detect_type_from_bytes(b"Plain text content")
        self.assertEqual(result, "txt")


class TestContextAnalyzer(unittest.TestCase):
    def setUp(self):
        # Используем маленькую модель для тестов
        self.analyzer = ContextAnalyzer(
            model_name="DeepPavlov/rubert-base-cased",
            max_length=64,
            feature_dim=768,
        )
        self.analyzer.eval()

    def test_encode_returns_correct_shape(self):
        texts = ["Проверь маркировку", "Проверь упаковку"]
        with torch.no_grad():
            emb, tokens = self.analyzer.encode(texts)
        self.assertEqual(emb.shape[0], 2)
        self.assertEqual(emb.shape[1], 768)

    def test_forward_returns_context_output(self):
        with torch.no_grad():
            out = self.analyzer(
                query=["Проверь соответствие маркировки"],
                document=["ГОСТ 123-2020. Требования к маркировке продуктов питания."],
            )
        self.assertEqual(out.context_vector.shape, (1, 768))
        self.assertEqual(out.query_embedding.shape, (1, 768))
        self.assertEqual(out.doc_embedding.shape, (1, 768))

    def test_with_history(self):
        with torch.no_grad():
            out = self.analyzer(
                query=["Проверь цвет"],
                document=["Стандарт цвета: красный."],
                history=["Предыдущий запрос: проверь форму."],
            )
        self.assertIsNotNone(out.context_vector)


if __name__ == "__main__":
    unittest.main()
