"""
YOLO-based object detector + 3×3 grid splitter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class DetectionBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float        # YOLO detection confidence
    class_id: int
    class_name: str
    crop: Image.Image
    region_name: str         # e.g. "yolo_0", "yolo_1", ...


class YOLODetector:
    GRID_ZONES = [
        "top-left", "top-center", "top-right",
        "center-left", "center", "center-right",
        "bottom-left", "bottom-center", "bottom-right",
    ]

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        max_detections: int = 20,
    ):
        self.conf = conf_threshold
        self.max_detections = max_detections
        self.device = device or "cpu"
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLO
            return YOLO(model_name)
        except Exception:
            # fallback for ultralytics <8.3 without yolo11n.pt
            from ultralytics import YOLO
            return YOLO("yolov8n.pt")

    def detect(self, image: Image.Image) -> list[DetectionBox]:
        results = self.model(image, conf=self.conf, verbose=False)
        boxes: list[DetectionBox] = []
        for r in results:
            names = r.names
            for i, box in enumerate(r.boxes):
                if i >= self.max_detections:
                    break
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))
                crop = image.crop((x1, y1, x2, y2))
                boxes.append(DetectionBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    crop=crop,
                    region_name=f"yolo_{i}",
                ))
        # sort by confidence descending
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes

    def grid_crops(self, image: Image.Image) -> list[tuple[str, Image.Image]]:
        """Return 9 grid crops + the full image (10 total)."""
        w, h = image.size
        crops: list[tuple[str, Image.Image]] = []
        for row in range(3):
            for col in range(3):
                x0 = col * w // 3
                y0 = row * h // 3
                x1 = (col + 1) * w // 3
                y1 = (row + 1) * h // 3
                zone_name = self.GRID_ZONES[row * 3 + col]
                crops.append((zone_name, image.crop((x0, y0, x1, y1))))
        crops.append(("full", image.copy()))
        return crops

    def get_all_crops(
        self, image: Image.Image
    ) -> tuple[list[DetectionBox], list[tuple[str, Image.Image]]]:
        """Convenience: detect() + grid_crops()."""
        boxes = self.detect(image)
        grid = self.grid_crops(image)
        return boxes, grid
