"""Предобработка изображений для визуального модуля."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImagePreprocessor:
    """
    Предобрабатывает изображения перед подачей в бэкбон.

    Поддерживает нормализацию ImageNet и аугментацию для обучения.
    """

    # Статистика ImageNet
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, image_size: int = 224, augment: bool = False):
        """
        Args:
            image_size: Целевой размер изображения (квадрат).
            augment: Применять аугментацию (для обучения).
        """
        self.image_size = image_size
        self.augment = augment
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        ops = []

        if self.augment:
            ops.extend([
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomRotation(15),
            ])
        else:
            ops.extend([
                transforms.Resize(int(self.image_size * 1.15)),
                transforms.CenterCrop(self.image_size),
            ])

        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        return transforms.Compose(ops)

    def load_image(self, source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """
        Загрузить изображение из различных источников.

        Args:
            source: Путь к файлу, байты или объект PIL.Image.

        Returns:
            Изображение в формате RGB PIL.Image.
        """
        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, (str, Path)):
            return Image.open(source).convert("RGB")

        if isinstance(source, bytes):
            import io
            return Image.open(io.BytesIO(source)).convert("RGB")

        raise TypeError(f"Неподдерживаемый тип источника изображения: {type(source)}")

    def preprocess(
        self,
        source: Union[str, Path, bytes, Image.Image],
    ) -> torch.Tensor:
        """
        Загрузить и преобразовать изображение в тензор.

        Args:
            source: Источник изображения.

        Returns:
            Тензор формы [1, C, H, W] с нормализацией ImageNet.
        """
        img = self.load_image(source)
        tensor = self.transform(img)          # [C, H, W]
        return tensor.unsqueeze(0)            # [1, C, H, W]

    def preprocess_batch(
        self,
        sources: list[Union[str, Path, bytes, Image.Image]],
    ) -> torch.Tensor:
        """
        Предобработать батч изображений.

        Args:
            sources: Список источников изображений.

        Returns:
            Тензор формы [B, C, H, W].
        """
        tensors = [self.preprocess(s).squeeze(0) for s in sources]
        return torch.stack(tensors, dim=0)

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Обратная нормализация тензора → numpy массив [H, W, 3] uint8.

        Args:
            tensor: Тензор формы [C, H, W] или [1, C, H, W].

        Returns:
            Изображение uint8 [H, W, 3].
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        img = tensor.cpu() * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        return (img * 255).astype(np.uint8)
