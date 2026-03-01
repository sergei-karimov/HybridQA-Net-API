"""
Главный инференс-pipeline HybridQA-Net.

Объединяет все четыре модуля в единый поток обработки:
1. Предобработка изображения
2. Визуальный анализ + Grad-CAM++
3. Семантический анализ текста + документа
4. Cross-attention слияние + классификация
5. Генерация отчёта
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .cache import CacheManager
from .fusion.cross_attention import CrossAttentionFusion
from .fusion.decision_head import DecisionHead
from .nlp.context_analyzer import ContextAnalyzer
from .nlp.document_parser import DocumentParser
from .report.generator import ReportGenerator
from .vision.backbone import VisionBackbone
from .vision.gradcam import GradCAMPlusPlus
from .vision.preprocessor import ImagePreprocessor
from utils.helpers import compute_hash, compute_text_hash, get_device, load_config
from utils.logger import setup_logger

logger = setup_logger("hybridqa.pipeline")


@dataclass
class AnalysisResult:
    """Полный результат анализа."""

    # Основные результаты
    label: int                          # 0 = не соответствует, 1 = соответствует
    verdict: str                        # Текстовый вердикт
    confidence: float                   # Уверенность [0, 1]
    defect_score: float                 # Вероятность дефекта [0, 1]

    # Отчёт
    report: str                         # Полный текстовый отчёт
    summary: str                        # Краткое резюме
    defects: list[str]                  # Список дефектов
    recommendations: list[str]          # Рекомендации

    # Визуализация
    attention_map: Optional[np.ndarray] = None    # Тепловая карта [H, W]
    overlay_image: Optional[np.ndarray] = None    # Оригинал + тепловая карта
    defect_mask: Optional[np.ndarray] = None      # Маска дефектов [H, W]
    attention_map_b64: str = ""                    # Base64 encoded PNG для API

    # Мета-информация
    processing_time_s: float = 0.0
    cached: bool = False
    metadata: dict = field(default_factory=dict)


class HybridQANet:
    """
    Главный класс Hybrid QA-Net — мультимодальная система контроля качества.

    Использование::

        system = HybridQANet()
        result = system.analyze(
            image="product_photo.jpg",
            standard_doc="quality_standard.pdf",
            query="Проверь соответствие маркировки стандарту",
        )
        print(result.report)
        print(result.confidence)
    """

    VERDICTS = {0: "Не соответствует стандарту", 1: "Соответствует стандарту"}

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Инициализация всех модулей системы.

        Args:
            config_path: Путь к файлу конфигурации.
        """
        logger.info("Инициализация HybridQA-Net...")
        self.config = load_config(config_path)
        self.device = get_device(self.config)
        logger.info(f"Устройство: {self.device}")

        # ---------------------------------------------------------- Модули
        vis_cfg = self.config["model"]["vision"]
        nlp_cfg = self.config["model"]["nlp"]
        fus_cfg = self.config["model"]["fusion"]
        rep_cfg = self.config["model"]["report"]

        # 1. Визуальный модуль
        logger.info(f"Загрузка визуального бэкбона: {vis_cfg['backbone']}")
        self.preprocessor = ImagePreprocessor(
            image_size=vis_cfg["image_size"]
        )
        self.vision = VisionBackbone(
            backbone_name=vis_cfg["backbone"],
            pretrained=vis_cfg["pretrained"],
            feature_dim=vis_cfg["feature_dim"],
        ).to(self.device)
        self.gradcam = GradCAMPlusPlus(
            target_layer=self.vision.model.blocks[-1]
            if self.vision.is_vit
            else list(self.vision.model.children())[-2]
        )

        # 2. Семантический модуль
        logger.info(f"Загрузка NLP модели: {nlp_cfg['model_name']}")
        self.context_analyzer = ContextAnalyzer(
            model_name=nlp_cfg["model_name"],
            max_length=nlp_cfg["max_length"],
            feature_dim=nlp_cfg["feature_dim"],
            device=self.device,
        ).to(self.device)

        # 3. Модуль слияния и принятия решений
        logger.info("Инициализация модуля слияния...")
        self.fusion = CrossAttentionFusion(
            embed_dim=fus_cfg["embed_dim"],
            num_heads=fus_cfg["num_heads"],
            num_layers=fus_cfg["num_layers"],
            dropout=fus_cfg["dropout"],
        ).to(self.device)
        self.decision_head = DecisionHead(
            input_dim=fus_cfg["embed_dim"],
            num_classes=fus_cfg["num_classes"],
        ).to(self.device)

        # 4. Генератор отчётов
        logger.info(f"Загрузка генератора отчётов: {rep_cfg['model_name']}")
        self.report_gen = ReportGenerator(
            model_name=rep_cfg["model_name"],
            max_input_length=rep_cfg["max_input_length"],
            max_output_length=rep_cfg["max_output_length"],
            num_beams=rep_cfg["num_beams"],
            device=self.device,
        )

        # Парсер документов
        self.doc_parser = DocumentParser()

        # Кэш
        self.cache = CacheManager(self.config.get("cache", {}))

        # Переводим модули в eval-режим
        self.vision.eval()
        self.context_analyzer.eval()
        self.fusion.eval()
        self.decision_head.eval()

        logger.info("HybridQA-Net успешно инициализирован.")

    # ================================================================ API
    def analyze(
        self,
        image: Union[str, Path, bytes, Image.Image],
        standard_doc: Union[str, Path, bytes] = "",
        query: str = "Проверь соответствие продукта стандарту качества",
        history: Optional[list[str]] = None,
        defect_threshold: float = 0.5,
        use_cache: bool = True,
    ) -> AnalysisResult:
        """
        Выполнить полный анализ соответствия продукта стандарту.

        Args:
            image: Изображение продукта.
            standard_doc: Документ стандарта качества (PDF или TXT).
            query: Запрос на проверку.
            history: История предыдущих запросов (для контекста).
            defect_threshold: Порог для определения дефектов.
            use_cache: Использовать кэш результатов.

        Returns:
            Полный результат анализа.
        """
        start_time = time.perf_counter()

        # Формируем ключ кэша
        if use_cache:
            cache_key = self._make_cache_key(image, standard_doc, query)
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached.cached = True
                logger.debug(f"Результат из кэша: {cache_key[:16]}...")
                return cached

        # ------------------------------------------------- Шаг 1: Изображение
        logger.debug("Предобработка изображения...")
        pixel_values = self.preprocessor.preprocess(image).to(self.device)
        original_image = self.preprocessor.load_image(image)
        orig_array = np.array(original_image)

        # ------------------------------------------------- Шаг 2: Документ
        doc_text = ""
        if standard_doc:
            logger.debug("Парсинг документа стандарта...")
            try:
                if isinstance(standard_doc, str) and self._is_raw_text(standard_doc):
                    doc_text = standard_doc  # Уже сырой текст, не путь к файлу
                else:
                    doc_text = self.doc_parser.parse(standard_doc)
            except Exception as e:
                logger.warning(f"Не удалось распарсить документ: {e}")
                doc_text = str(standard_doc) if isinstance(standard_doc, str) else ""

        doc_text = doc_text or "Стандарт качества не предоставлен."

        # ------------------------------------------------- Шаг 3: Visual forward
        logger.debug("Визуальный анализ...")
        with torch.no_grad():
            vision_out = self.vision(pixel_values)

        # ------------------------------------------------- Шаг 4: NLP forward
        logger.debug("Семантический анализ...")
        with torch.no_grad():
            context_out = self.context_analyzer(
                query=[query],
                document=[doc_text[:self.context_analyzer.max_length]],
                history=history,
            )

        # ------------------------------------------------- Шаг 5: Fusion
        logger.debug("Слияние признаков...")
        with torch.no_grad():
            patch_feats = vision_out.patch_features   # [1, N, D]
            text_tokens = context_out.token_embeddings[:, :, :]  # [1, L, D]

            # Проецируем в одинаковое пространство если нужно
            fusion_out = self.fusion(patch_feats, text_tokens)
            decision_out = self.decision_head(fusion_out.fused_embedding)

        # ------------------------------------------------- Шаг 6: Grad-CAM++
        logger.debug("Генерация карты внимания (Grad-CAM++)...")
        attention_map, overlay, defect_mask = self._compute_attention(
            pixel_values=pixel_values,
            vision_output=vision_out,
            original_image=orig_array,
            threshold=defect_threshold,
        )

        # ------------------------------------------------- Шаг 7: Дефекты
        defects = self._extract_defects(
            defect_mask=defect_mask,
            attention_map=attention_map,
            query=query,
            decision_out=decision_out,
        )

        # ------------------------------------------------- Шаг 8: Отчёт
        logger.debug("Генерация отчёта...")
        verdict_str = self.VERDICTS[decision_out.label]
        report_out = self.report_gen.generate(
            query=query,
            verdict=verdict_str,
            confidence=decision_out.confidence,
            defects=defects,
            doc_excerpt=doc_text[:500],
        )

        # ------------------------------------------------- Шаг 9: Base64 изображение
        attn_b64 = ""
        if overlay is not None:
            attn_b64 = self._array_to_base64(overlay)

        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Анализ завершён за {elapsed:.2f}с | "
            f"Вердикт: {verdict_str} | "
            f"Уверенность: {decision_out.confidence:.1%}"
        )

        result = AnalysisResult(
            label=decision_out.label,
            verdict=verdict_str,
            confidence=decision_out.confidence,
            defect_score=decision_out.defect_score,
            report=report_out.full_report,
            summary=report_out.summary,
            defects=report_out.defects,
            recommendations=report_out.recommendations,
            attention_map=attention_map,
            overlay_image=overlay,
            defect_mask=defect_mask,
            attention_map_b64=attn_b64,
            processing_time_s=elapsed,
            cached=False,
        )

        # Сохраняем в кэш
        if use_cache:
            self.cache.set(cache_key, result)

        return result

    def analyze_batch(
        self,
        items: list[dict],
        use_cache: bool = True,
    ) -> list[AnalysisResult]:
        """
        Пакетная обработка нескольких запросов.

        Args:
            items: Список словарей с ключами image, standard_doc, query.
            use_cache: Использовать кэш.

        Returns:
            Список AnalysisResult.
        """
        return [
            self.analyze(
                image=item["image"],
                standard_doc=item.get("standard_doc", ""),
                query=item.get("query", "Проверь соответствие"),
                history=item.get("history"),
                use_cache=use_cache,
            )
            for item in items
        ]

    def analyze_conditions(
        self,
        image: Union[str, Path, bytes, Image.Image],
        conditions: list[dict],
        standard_doc: Union[str, Path, bytes] = "",
        use_cache: bool = True,
    ) -> dict:
        """
        Проверить несколько условий для одного изображения за один вызов.

        Args:
            image: Изображение продукта.
            conditions: Список словарей с ключами id, query, type ("must"|"must_not").
            standard_doc: Документ стандарта качества.
            use_cache: Использовать кэш.

        Returns:
            Словарь с overall_pass, counts и results по каждому условию.
        """
        start = time.perf_counter()
        results = []
        for cond in conditions:
            result = self.analyze(
                image=image,
                standard_doc=standard_doc,
                query=cond["query"],
                use_cache=use_cache,
            )
            cond_type = cond.get("type", "must")
            passed = (result.label == 1) if cond_type == "must" else (result.label == 0)
            results.append({
                "id": cond["id"],
                "query": cond["query"],
                "type": cond_type,
                "passed": passed,
                "confidence": result.confidence,
                "attention_map_b64": result.attention_map_b64 or None,
            })
        passed_count = sum(1 for r in results if r["passed"])
        return {
            "overall_pass": passed_count == len(results),
            "conditions_checked": len(results),
            "conditions_passed": passed_count,
            "conditions_failed": len(results) - passed_count,
            "results": results,
            "total_processing_time_s": time.perf_counter() - start,
        }

    @staticmethod
    def _is_raw_text(s: str) -> bool:
        """
        Эвристика: является ли строка текстом (не путём к файлу).
        Признаки текста: содержит пробелы/переносы, длиннее 260 символов,
        или начинается не с '/' или './' и не похоже на путь.
        """
        if "\n" in s or len(s) > 260:
            return True
        # Разрешённые расширения файлов
        file_extensions = (".pdf", ".txt", ".text", ".docx")
        return not any(s.lower().endswith(ext) for ext in file_extensions)

    # ============================================================= Helpers
    def _compute_attention(
        self,
        pixel_values: torch.Tensor,
        vision_output,
        original_image: np.ndarray,
        threshold: float,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Вычислить карту внимания через Grad-CAM++."""
        try:
            # Для Grad-CAM нужен backward pass через logits
            # Используем surrogate backward через визуальные эмбеддинги
            pixel_values_grad = pixel_values.clone().requires_grad_(True)

            with torch.enable_grad():
                vision_out_g = self.vision(pixel_values_grad)
                # Суррогатный скалярный выход для backward
                proxy_output = vision_out_g.embeddings.sum(dim=-1, keepdim=True)
                # Добавляем размерность num_classes для GradCAM
                proxy_logits = proxy_output.expand(-1, 2)

            heatmap = self.gradcam.generate(
                model_output=proxy_logits,
                target_class=1,
            )

            overlay = self.gradcam.overlay_on_image(original_image, heatmap)
            mask = self.gradcam.generate_defect_mask(heatmap, threshold)

            return heatmap, overlay, mask

        except Exception as e:
            logger.warning(f"Grad-CAM++ не удался: {e}")
            return None, None, None

    @staticmethod
    def _extract_defects(
        defect_mask,
        attention_map,
        query: str,
        decision_out,
    ) -> list[str]:
        """Сформировать список текстовых описаний дефектов."""
        defects = []

        if decision_out.label == 0:  # Не соответствует
            if defect_mask is not None:
                coverage = float(defect_mask.mean()) * 100
                if coverage > 30:
                    defects.append(
                        f"Значительная область несоответствия: {coverage:.1f}% изображения"
                    )
                elif coverage > 10:
                    defects.append(
                        f"Локальная область несоответствия: {coverage:.1f}% изображения"
                    )
                else:
                    defects.append("Точечное несоответствие в области с высокой активацией")

            defects.append(
                f"Обнаружено несоответствие по запросу: «{query[:100]}»"
            )

        return defects

    @staticmethod
    def _make_cache_key(image, standard_doc, query: str) -> str:
        """Создать ключ кэша из входных данных."""
        parts = []

        if isinstance(image, bytes):
            parts.append(compute_hash(image))
        elif isinstance(image, (str, Path)):
            parts.append(compute_text_hash(str(image)))
        else:
            parts.append("pil_image")

        if isinstance(standard_doc, bytes):
            parts.append(compute_hash(standard_doc))
        elif standard_doc:
            parts.append(compute_text_hash(str(standard_doc)))

        parts.append(compute_text_hash(query))
        return "_".join(parts)

    @staticmethod
    def _array_to_base64(array: np.ndarray) -> str:
        """Конвертировать numpy массив в base64 PNG строку."""
        img = Image.fromarray(array.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # ============================================================ Training
    def fine_tune_prepare(
        self,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 2,
    ) -> None:
        """
        Подготовить модель к fine-tuning.

        Args:
            freeze_backbone: Заморозить бэкбон (обучается только голова).
            unfreeze_last_n: Разморозить последние N блоков бэкбона.
        """
        if freeze_backbone:
            self.vision.freeze_backbone()
            self.context_analyzer.freeze()
        else:
            self.vision.unfreeze_backbone(last_n_blocks=unfreeze_last_n)
            self.context_analyzer.unfreeze_last_layers(n_layers=unfreeze_last_n)

        # Голова принятия решений всегда обучается
        for param in self.fusion.parameters():
            param.requires_grad = True
        for param in self.decision_head.parameters():
            param.requires_grad = True

        # Переводим в train mode
        self.vision.train()
        self.context_analyzer.train()
        self.fusion.train()
        self.decision_head.train()

        trainable = sum(p.numel() for p in self._all_params() if p.requires_grad)
        total = sum(p.numel() for p in self._all_params())
        logger.info(f"Параметры для обучения: {trainable:,} / {total:,}")

    def _all_params(self):
        """Все параметры системы."""
        yield from self.vision.parameters()
        yield from self.context_analyzer.parameters()
        yield from self.fusion.parameters()
        yield from self.decision_head.parameters()

    def save_checkpoint(self, path: str) -> None:
        """Сохранить checkpoint системы."""
        torch.save(
            {
                "vision": self.vision.state_dict(),
                "context_analyzer": self.context_analyzer.state_dict(),
                "fusion": self.fusion.state_dict(),
                "decision_head": self.decision_head.state_dict(),
                "config": self.config,
            },
            path,
        )
        logger.info(f"Checkpoint сохранён: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Загрузить checkpoint системы."""
        checkpoint = torch.load(path, map_location=self.device)
        self.vision.load_state_dict(checkpoint["vision"])
        self.context_analyzer.load_state_dict(checkpoint["context_analyzer"])
        self.fusion.load_state_dict(checkpoint["fusion"])
        self.decision_head.load_state_dict(checkpoint["decision_head"])
        logger.info(f"Checkpoint загружен: {path}")
