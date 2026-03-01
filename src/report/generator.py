"""
Генератор структурированных отчётов на базе T5.

Использует ``cointegrated/rut5-base`` — русскоязычная T5-модель.
Формирует детальный отчёт о соответствии/несоответствии продукта стандарту.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


@dataclass
class ReportOutput:
    """Структурированный отчёт."""

    full_report: str               # Полный текстовый отчёт
    summary: str                   # Краткое резюме
    defects: list[str]             # Список обнаруженных дефектов
    recommendations: list[str]     # Рекомендации


class ReportGenerator:
    """
    Генератор структурированных отчётов о качестве на базе T5.

    Принимает контекст анализа (запрос, результат проверки, дефекты)
    и генерирует связный текстовый отчёт на русском языке.
    """

    # Шаблон промпта для генерации отчёта
    PROMPT_TEMPLATE = (
        "Составь отчёт о контроле качества.\n"
        "Запрос: {query}\n"
        "Результат проверки: {verdict}\n"
        "Уверенность: {confidence:.1%}\n"
        "Обнаруженные области: {defect_info}\n"
        "Документ стандарта: {doc_excerpt}\n"
        "Отчёт:"
    )

    def __init__(
        self,
        model_name: str = "cointegrated/rut5-base",
        max_input_length: int = 1024,
        max_output_length: int = 512,
        num_beams: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace модель T5.
            max_input_length: Максимальная длина входного промпта.
            max_output_length: Максимальная длина генерируемого текста.
            num_beams: Число лучей для beam search.
            device: Устройство для вычислений.
        """
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.device = device or torch.device("cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        query: str,
        verdict: str,
        confidence: float,
        defects: list[str],
        doc_excerpt: str = "",
    ) -> ReportOutput:
        """
        Сгенерировать отчёт.

        Args:
            query: Исходный запрос пользователя.
            verdict: "Соответствует" или "Не соответствует".
            confidence: Уверенность модели [0, 1].
            defects: Список описаний дефектов/областей несоответствия.
            doc_excerpt: Отрывок из документа стандарта.

        Returns:
            Структурированный ReportOutput.
        """
        defect_info = (
            "; ".join(defects) if defects else "дефекты не обнаружены"
        )
        doc_excerpt_short = doc_excerpt[:300] if doc_excerpt else "не указан"

        prompt = self.PROMPT_TEMPLATE.format(
            query=query,
            verdict=verdict,
            confidence=confidence,
            defect_info=defect_info,
            doc_excerpt=doc_excerpt_short,
        )

        # Токенизация
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Генерация текста
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                do_sample=False,
            )

        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        return self._parse_report(generated_text, verdict, defects, confidence)

    @staticmethod
    def _parse_report(
        text: str,
        verdict: str,
        defects: list[str],
        confidence: float,
    ) -> ReportOutput:
        """
        Структурировать сгенерированный текст в ReportOutput.

        Args:
            text: Сгенерированный текст.
            verdict: Вердикт классификатора.
            defects: Список дефектов.
            confidence: Уверенность.

        Returns:
            Структурированный ReportOutput.
        """
        # Формируем полный отчёт с заголовками
        full_report = (
            f"=== ОТЧЁТ О КОНТРОЛЕ КАЧЕСТВА ===\n\n"
            f"Вердикт: {verdict}\n"
            f"Уверенность модели: {confidence:.1%}\n\n"
            f"--- Анализ ---\n{text}\n\n"
        )

        if defects:
            full_report += "--- Обнаруженные несоответствия ---\n"
            for i, d in enumerate(defects, 1):
                full_report += f"  {i}. {d}\n"
        else:
            full_report += "--- Несоответствий не обнаружено ---\n"

        # Рекомендации на основе вердикта
        recommendations = _build_recommendations(verdict, defects)

        return ReportOutput(
            full_report=full_report,
            summary=f"{verdict} (уверенность: {confidence:.1%})",
            defects=defects,
            recommendations=recommendations,
        )

    def generate_batch(
        self,
        items: list[dict],
    ) -> list[ReportOutput]:
        """
        Генерировать отчёты для батча.

        Args:
            items: Список словарей с ключами query, verdict, confidence, defects, doc_excerpt.

        Returns:
            Список ReportOutput.
        """
        return [
            self.generate(
                query=item.get("query", ""),
                verdict=item.get("verdict", ""),
                confidence=item.get("confidence", 0.0),
                defects=item.get("defects", []),
                doc_excerpt=item.get("doc_excerpt", ""),
            )
            for item in items
        ]


def _build_recommendations(verdict: str, defects: list[str]) -> list[str]:
    """Формирование рекомендаций на основе вердикта и дефектов."""
    if "соответствует" in verdict.lower() and "не" not in verdict.lower():
        return [
            "Продукт соответствует стандарту. Дополнительных действий не требуется.",
            "Рекомендуется сохранить отчёт для документирования.",
        ]

    recs = [
        "Провести детальный осмотр выявленных областей несоответствия.",
        "Задокументировать отклонения и уведомить ответственного специалиста.",
    ]

    if defects:
        recs.append(
            f"Устранить следующие несоответствия перед повторной проверкой: "
            f"{', '.join(defects[:3])}."
        )

    recs.append("Провести повторную проверку после устранения замечаний.")
    return recs
