"""Тесты визуального модуля."""

import io
import unittest

import numpy as np
import torch
from PIL import Image

from src.vision.backbone import VisionBackbone
from src.vision.gradcam import GradCAMPlusPlus
from src.vision.preprocessor import ImagePreprocessor


def _make_dummy_image(h: int = 224, w: int = 224) -> Image.Image:
    """Создать фиктивное цветное изображение."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = ImagePreprocessor(image_size=224)
        self.dummy_img = _make_dummy_image()

    def test_preprocess_returns_correct_shape(self):
        tensor = self.preprocessor.preprocess(self.dummy_img)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_preprocess_batch(self):
        imgs = [_make_dummy_image() for _ in range(4)]
        batch = self.preprocessor.preprocess_batch(imgs)
        self.assertEqual(batch.shape, (4, 3, 224, 224))

    def test_denormalize_returns_uint8(self):
        tensor = self.preprocessor.preprocess(self.dummy_img)
        arr = self.preprocessor.denormalize(tensor)
        self.assertEqual(arr.dtype, np.uint8)
        self.assertEqual(arr.shape, (224, 224, 3))

    def test_load_from_bytes(self):
        buf = io.BytesIO()
        self.dummy_img.save(buf, format="JPEG")
        tensor = self.preprocessor.preprocess(buf.getvalue())
        self.assertEqual(tensor.shape, (1, 3, 224, 224))


class TestVisionBackbone(unittest.TestCase):
    def setUp(self):
        self.backbone = VisionBackbone(
            backbone_name="vit_base_patch16_224",
            pretrained=False,   # Без загрузки весов для ускорения тестов
            feature_dim=768,
        )
        self.backbone.eval()
        self.preprocessor = ImagePreprocessor(image_size=224)
        self.dummy_tensor = self.preprocessor.preprocess(_make_dummy_image())

    def test_output_shapes(self):
        with torch.no_grad():
            out = self.backbone(self.dummy_tensor)

        self.assertEqual(out.embeddings.shape, (1, 768))
        self.assertEqual(out.patch_features.shape[0], 1)
        self.assertGreater(out.patch_features.shape[1], 0)

    def test_freeze_backbone(self):
        self.backbone.freeze_backbone()
        for param in self.backbone.model.parameters():
            self.assertFalse(param.requires_grad)

    def test_unfreeze_last_blocks(self):
        self.backbone.freeze_backbone()
        self.backbone.unfreeze_backbone(last_n_blocks=1)
        # Проекция должна быть разморожена
        for param in self.backbone.proj.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
