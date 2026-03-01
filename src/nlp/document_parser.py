"""
Парсер текстовых документов (PDF, TXT).

Извлекает структурированный текст из документов для анализа контекста.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union


class DocumentParser:
    """
    Парсер документов: PDF и TXT.

    Пример использования::

        parser = DocumentParser()
        text = parser.parse("quality_standard.pdf")
    """

    def parse(
        self,
        source: Union[str, Path, bytes],
        file_type: str = "auto",
    ) -> str:
        """
        Извлечь текст из документа.

        Args:
            source: Путь к файлу, байты или строка текста.
            file_type: "pdf", "txt" или "auto" (определяется автоматически).

        Returns:
            Очищенный текст документа.
        """
        if isinstance(source, bytes):
            resolved_type = self._detect_type_from_bytes(source) if file_type == "auto" else file_type
            return self._parse_bytes(source, resolved_type)

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {source}")

        resolved_type = path.suffix.lower().lstrip(".") if file_type == "auto" else file_type

        if resolved_type == "pdf":
            return self._parse_pdf(path)
        elif resolved_type in ("txt", "text"):
            return self._parse_txt(path)
        else:
            raise ValueError(f"Неподдерживаемый тип файла: '{resolved_type}'")

    # ----------------------------------------------------------------- PDF
    def _parse_pdf(self, path: Path) -> str:
        """Извлечь текст из PDF с помощью PyPDF2."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return self._clean_text("\n\n".join(pages))
        except ImportError:
            raise ImportError(
                "Для парсинга PDF установите pypdf: pip install pypdf"
            )

    def _parse_txt(self, path: Path) -> str:
        """Прочитать TXT-файл с автоопределением кодировки."""
        encodings = ("utf-8", "cp1251", "latin-1")
        for enc in encodings:
            try:
                text = path.read_text(encoding=enc)
                return self._clean_text(text)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Не удалось определить кодировку файла: {path}")

    def _parse_bytes(self, data: bytes, file_type: str) -> str:
        """Парсить документ из байт."""
        if file_type == "pdf":
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(data))
                pages = [p.extract_text() or "" for p in reader.pages]
                return self._clean_text("\n\n".join(pages))
            except ImportError:
                raise ImportError("Установите pypdf: pip install pypdf")
        else:
            for enc in ("utf-8", "cp1251", "latin-1"):
                try:
                    return self._clean_text(data.decode(enc))
                except UnicodeDecodeError:
                    continue
            raise ValueError("Не удалось декодировать байты как текст")

    @staticmethod
    def _detect_type_from_bytes(data: bytes) -> str:
        """Определить тип файла по сигнатуре байт."""
        if data[:4] == b"%PDF":
            return "pdf"
        return "txt"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Очистить текст: убрать лишние пробелы, спецсимволы."""
        # Нормализация переносов строк
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\r", "\n", text)
        # Убрать повторяющиеся пробелы и табуляции
        text = re.sub(r"[ \t]+", " ", text)
        # Убрать пробелы перед переносом строки (trailing whitespace на каждой строке)
        text = re.sub(r" +\n", "\n", text)
        # Убрать более двух пустых строк подряд
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def parse_to_chunks(
        self,
        source: Union[str, Path, bytes],
        chunk_size: int = 512,
        overlap: int = 64,
        file_type: str = "auto",
    ) -> list[str]:
        """
        Разбить документ на перекрывающиеся чанки для длинных текстов.

        Args:
            source: Источник документа.
            chunk_size: Размер чанка в символах.
            overlap: Перекрытие между чанками.
            file_type: Тип файла.

        Returns:
            Список текстовых чанков.
        """
        text = self.parse(source, file_type)
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks
