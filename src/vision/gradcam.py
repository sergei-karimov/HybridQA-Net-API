"""
Реализация Grad-CAM++ для визуализации областей внимания модели.

Grad-CAM++ — улучшение Grad-CAM с лучшей локализацией для нескольких объектов.
Ссылка: Chattopadhay et al., 2018 (https://arxiv.org/abs/1710.11063)
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAMPlusPlus:
    """
    Grad-CAM++ для произвольного слоя модели.

    Пример использования::

        cam = GradCAMPlusPlus(model.model.blocks[-1])
        heatmap = cam.generate(pixel_values, target_class=1)
        overlay = cam.overlay_on_image(original_image, heatmap)
    """

    def __init__(self, target_layer: torch.nn.Module):
        """
        Args:
            target_layer: Слой модели, для которого генерировать карту внимания.
        """
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register()

    def _register(self) -> None:
        """Регистрация forward и backward хуков."""

        def forward_hook(module, input, output):
            # Для ViT output: [B, N+1, D]; для CNN: [B, C, H, W]
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(
        self,
        model_output: torch.Tensor,
        target_class: Optional[int] = None,
        spatial_size: tuple[int, int] = (14, 14),
    ) -> np.ndarray:
        """
        Сгенерировать тепловую карту Grad-CAM++.

        Args:
            model_output: Логиты или эмбеддинги для backward pass.
            target_class: Целевой класс (None → argmax).
            spatial_size: Размер пространственной карты (H, W).

        Returns:
            Нормализованная тепловая карта [H, W] в диапазоне [0, 1].
        """
        if target_class is None:
            target_class = model_output.argmax(dim=-1).item()

        # Backward pass
        model_output[0, target_class].backward(retain_graph=True)

        grads = self._gradients    # [B, N, D] или [B, C, H, W]
        acts = self._activations   # аналогично

        if grads is None or acts is None:
            raise RuntimeError("Градиенты не получены. Убедитесь, что backward выполнен.")

        # ViT: преобразуем патч-токены в пространственную карту
        if acts.dim() == 3:
            acts, grads = self._vit_to_spatial(acts, grads, spatial_size)

        # Grad-CAM++ веса
        weights = self._compute_gradcampp_weights(grads, acts)

        # Взвешенная комбинация карт активаций
        cam = (weights * acts).sum(dim=1, keepdim=True)   # [B, 1, H, W]
        cam = F.relu(cam)                                  # Только положительные активации

        # Нормализация в [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.astype(np.float32)

    @staticmethod
    def _vit_to_spatial(
        acts: torch.Tensor,
        grads: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Преобразовать ViT [B, N, D] → [B, D, H, W]."""
        B, N, D = acts.shape
        H, W = spatial_size

        # Убираем CLS-токен если есть
        if N == H * W + 1:
            acts = acts[:, 1:, :]
            grads = grads[:, 1:, :]

        acts = acts.permute(0, 2, 1).reshape(B, D, H, W)
        grads = grads.permute(0, 2, 1).reshape(B, D, H, W)
        return acts, grads

    @staticmethod
    def _compute_gradcampp_weights(
        grads: torch.Tensor,   # [B, C, H, W]
        acts: torch.Tensor,    # [B, C, H, W]
    ) -> torch.Tensor:
        """
        Вычислить веса Grad-CAM++ по формуле из статьи.

        Returns:
            Веса [B, C, 1, 1].
        """
        # Числитель: grad²
        alpha_num = grads ** 2

        # Знаменатель: 2*grad² + sum(acts * grad³)
        alpha_denom = (
            2.0 * grads ** 2
            + acts * (grads ** 3).sum(dim=(2, 3), keepdim=True)
        )
        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom),
        )

        alpha = alpha_num / (alpha_denom + 1e-8)
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        return weights

    def overlay_on_image(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Наложить тепловую карту на исходное изображение.

        Args:
            image: RGB изображение [H, W, 3] uint8.
            heatmap: Карта [H, W] float32 в диапазоне [0, 1].
            alpha: Прозрачность наложения.
            colormap: OpenCV colormap.

        Returns:
            Изображение с наложенной тепловой картой [H, W, 3] uint8.
        """
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
        return overlay

    def generate_defect_mask(
        self,
        heatmap: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Получить бинарную маску дефектов из тепловой карты.

        Args:
            heatmap: Карта [H, W] в диапазоне [0, 1].
            threshold: Порог бинаризации.

        Returns:
            Бинарная маска [H, W] dtype bool.
        """
        return heatmap >= threshold

    def remove_hooks(self) -> None:
        """Удалить зарегистрированные хуки."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __del__(self):
        self.remove_hooks()
