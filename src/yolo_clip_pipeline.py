"""
YOLO + Multilingual CLIP pipeline (Variant B).

Parallel pipeline to HybridQANet — no retraining required.
Handles Russian queries natively via clip-ViT-B-32-multilingual-v1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from src.vision.detector import DetectionBox, YOLODetector
from src.vision.clip_matcher import CLIPMatcher
from utils.helpers import compute_hash, compute_text_hash


@dataclass
class YoloCLIPResult:
    label: int                          # 0 = does not conform, 1 = conforms
    verdict: str
    confidence: float                   # == best_similarity
    threshold: float
    yolo_detections: list[dict]         # {region_name, class_name, x1,y1,x2,y2, yolo_confidence, clip_similarity}
    grid_regions: list[dict]            # {region_name, clip_similarity, crop_width, crop_height}
    best_region: str
    best_similarity: float
    all_similarities: list[dict]        # [{region_name, similarity}] sorted desc
    query: str
    normalized_query: str               # query after stripping normative constructions
    processing_time_s: float
    cached: bool = False


class YoloCLIPAnalyzer:
    VERDICTS = {0: "Не соответствует стандарту", 1: "Соответствует стандарту"}

    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        clip_image_model: str = "sentence-transformers/clip-ViT-B-32",
        clip_text_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        similarity_threshold: float = 0.25,
        yolo_conf_threshold: float = 0.25,
        max_yolo_detections: int = 20,
        device: Optional[str] = None,
    ):
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = similarity_threshold
        self.detector = YOLODetector(
            model_name=yolo_model,
            device=_device,
            conf_threshold=yolo_conf_threshold,
            max_detections=max_yolo_detections,
        )
        self.matcher = CLIPMatcher(
            image_model_name=clip_image_model,
            text_model_name=clip_text_model,
            device=_device,
        )
        self._cache: dict[str, YoloCLIPResult] = {}

    # ----------------------------------------------------------------- public
    def analyze(
        self,
        image: Union[bytes, str, Path, Image.Image],
        query: str,
        use_cache: bool = True,
    ) -> YoloCLIPResult:
        t0 = time.perf_counter()

        pil_image = self._load_image(image)

        if use_cache:
            key = self._cache_key(image, query)
            if key in self._cache:
                cached = self._cache[key]
                # return a copy with cached=True
                result = YoloCLIPResult(**{**cached.__dict__, "cached": True})
                return result

        norm_query = self._normalize_query(query)

        yolo_boxes, grid_pairs = self.detector.get_all_crops(pil_image)

        # Combine all crops: YOLO detections first, then grid zones
        all_crops: list[Image.Image] = []
        all_names: list[str] = []

        for box in yolo_boxes:
            all_crops.append(box.crop)
            all_names.append(box.region_name)

        for name, crop in grid_pairs:
            all_crops.append(crop)
            all_names.append(name)

        match_result = self.matcher.match(all_crops, all_names, norm_query)

        label = 1 if match_result.best_similarity >= self.threshold else 0
        verdict = self.VERDICTS[label]

        # Build per-region dicts
        region_by_name = {r.region_name: r.similarity for r in match_result.all_regions}

        yolo_dicts = [
            {
                "region_name": box.region_name,
                "class_name": box.class_name,
                "class_id": box.class_id,
                "x1": box.x1, "y1": box.y1,
                "x2": box.x2, "y2": box.y2,
                "yolo_confidence": box.confidence,
                "clip_similarity": region_by_name.get(box.region_name, 0.0),
            }
            for box in yolo_boxes
        ]

        grid_dicts = [
            {
                "region_name": name,
                "clip_similarity": region_by_name.get(name, 0.0),
                "crop_width": crop.width,
                "crop_height": crop.height,
            }
            for name, crop in grid_pairs
        ]

        all_sims = sorted(
            [{"region_name": r.region_name, "similarity": r.similarity}
             for r in match_result.all_regions],
            key=lambda x: x["similarity"],
            reverse=True,
        )

        elapsed = time.perf_counter() - t0

        result = YoloCLIPResult(
            label=label,
            verdict=verdict,
            confidence=match_result.best_similarity,
            threshold=self.threshold,
            yolo_detections=yolo_dicts,
            grid_regions=grid_dicts,
            best_region=match_result.best_region,
            best_similarity=match_result.best_similarity,
            all_similarities=all_sims,
            query=query,
            normalized_query=norm_query,
            processing_time_s=elapsed,
            cached=False,
        )

        if use_cache:
            self._cache[key] = result

        return result

    def analyze_conditions(
        self,
        image: Union[bytes, str, Path, Image.Image],
        conditions: list[dict],
        standard_doc: str = "",
        use_cache: bool = True,
    ) -> dict:
        """
        Same interface as HybridQANet.analyze_conditions().
        Each condition: {id, query, type} where type is "must" or "must_not".
        """
        t0 = time.perf_counter()
        pil_image = self._load_image(image)

        results = []
        passed_count = 0

        for cond in conditions:
            cond_id = cond.get("id", "")
            query = cond.get("query", "")
            cond_type = cond.get("type", "must")

            r = self.analyze(pil_image, query, use_cache=use_cache)

            if cond_type == "must":
                passed = r.label == 1
            else:  # must_not
                passed = r.label == 0

            if passed:
                passed_count += 1

            results.append({
                "id": cond_id,
                "query": query,
                "type": cond_type,
                "passed": passed,
                "confidence": r.confidence,
                "best_region": r.best_region,
                "best_similarity": r.best_similarity,
                "yolo_detections_count": len(r.yolo_detections),
            })

        total_elapsed = time.perf_counter() - t0
        failed_count = len(conditions) - passed_count

        return {
            "overall_pass": failed_count == 0,
            "conditions_checked": len(conditions),
            "conditions_passed": passed_count,
            "conditions_failed": failed_count,
            "results": results,
            "total_processing_time_s": total_elapsed,
        }

    # ----------------------------------------------------------------- static
    @staticmethod
    def _load_image(source: Union[bytes, str, Path, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")
        if isinstance(source, (bytes, bytearray)):
            import io
            return Image.open(io.BytesIO(source)).convert("RGB")
        raise TypeError(f"Unsupported image source type: {type(source)}")

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Strip Russian normative/modal constructions so CLIP gets a visual description.

        Examples:
            'Шахматный узор должен быть справа'       → 'Шахматный узор справа'
            'Логотип должен находиться в центре'       → 'Логотип в центре'
            'Шахматный узор должен отсутствовать'      → 'Шахматный узор'
            'Необходимо наличие синей полосы сверху'   → 'наличие синей полосы сверху'
        """
        import re
        patterns = [
            # "должен/должна/должно/должны" + following verb (infinitive)
            r'должен\s+\w+',
            r'должна\s+\w+',
            r'должно\s+\w+',
            r'должны\s+\w+',
            # standalone modal words (if no verb was matched)
            r'должен', r'должна', r'должно', r'должны',
            r'необходимо', r'нужно', r'следует', r'требуется', r'надо',
        ]
        result = query
        for p in patterns:
            result = re.sub(p, '', result, flags=re.IGNORECASE)
        result = re.sub(r'\s+', ' ', result).strip()
        return result or query  # fallback to original if everything was stripped

    @staticmethod
    def _cache_key(image: Union[bytes, str, Path, Image.Image], query: str) -> str:
        if isinstance(image, (bytes, bytearray)):
            img_hash = compute_hash(image)
        else:
            img_hash = compute_text_hash(str(image))
        return f"{img_hash}:{compute_text_hash(query)}"
