"""
Визуальный бэкбон: Vision Transformer (ViT) или EfficientNet-B7.

Возвращает эмбеддинги изображения и промежуточные карты признаков
для последующего Grad-CAM++ и слияния с текстовым модулем.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import timm


@dataclass
class VisionOutput:
    """Выход визуального бэкбона."""

    embeddings: torch.Tensor        # [B, D] — глобальный эмбеддинг
    patch_features: torch.Tensor    # [B, N, D] — патч-эмбеддинги (только ViT)
    spatial_features: torch.Tensor  # [B, D, H, W] — пространственная карта признаков
    logits: Optional[torch.Tensor]  # [B, num_classes] — опционально для fine-tuning


class VisionBackbone(nn.Module):
    """
    Унифицированный визуальный бэкбон на базе timm.

    Поддерживает:
    - Vision Transformer: ``vit_base_patch16_224``
    - EfficientNet-B7: ``efficientnet_b7``
    """

    SUPPORTED = {
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "efficientnet_b7",
        "efficientnet_b4",
    }

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_dim: int = 768,
        num_classes: int = 0,   # 0 → не создавать классификатор
    ):
        """
        Args:
            backbone_name: Имя модели в timm.
            pretrained: Загружать предобученные веса.
            feature_dim: Выходная размерность эмбеддинга.
            num_classes: Число выходных классов (0 = feature extractor).
        """
        super().__init__()

        if backbone_name not in self.SUPPORTED:
            raise ValueError(
                f"Неподдерживаемый бэкбон '{backbone_name}'. "
                f"Доступны: {self.SUPPORTED}"
            )

        self.backbone_name = backbone_name
        self.is_vit = backbone_name.startswith("vit")

        # Загрузка модели через timm
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # Убираем классификационную голову
            global_pool="avg" if not self.is_vit else "token",
        )

        # Определяем размерность признаков бэкбона
        backbone_dim = self.model.num_features

        # Проекция в нужную размерность (если отличается)
        self.proj = (
            nn.Linear(backbone_dim, feature_dim)
            if backbone_dim != feature_dim
            else nn.Identity()
        )

        # Опциональный классификатор для fine-tuning
        self.classifier: Optional[nn.Linear] = (
            nn.Linear(feature_dim, num_classes) if num_classes > 0 else None
        )

        self.feature_dim = feature_dim
        self._register_hooks()

    # ------------------------------------------------------------------ hooks
    def _register_hooks(self) -> None:
        """Регистрация forward-хуков для извлечения промежуточных признаков."""
        self._spatial_features: Optional[torch.Tensor] = None
        self._patch_features: Optional[torch.Tensor] = None

        if self.is_vit:
            # Последний блок ViT — для Grad-CAM и патч-признаков
            target_layer = self.model.blocks[-1]

            def vit_hook(module, input, output):
                # output: [B, N+1, D]  (первый токен — CLS)
                self._patch_features = output[:, 1:, :]    # без CLS [B, N, D]
                # Собираем пространственную карту (sqrt(N) × sqrt(N))
                B, N, D = self._patch_features.shape
                H = W = int(N ** 0.5)
                self._spatial_features = (
                    self._patch_features.permute(0, 2, 1)   # [B, D, N]
                    .reshape(B, D, H, W)                    # [B, D, H, W]
                    .contiguous()
                )

            target_layer.register_forward_hook(vit_hook)

        else:
            # EfficientNet — последний конволюционный блок
            target_layer = list(self.model.children())[-2]

            def cnn_hook(module, input, output):
                self._spatial_features = output  # [B, D, H, W]

            target_layer.register_forward_hook(cnn_hook)

    # -------------------------------------------------------------- forward
    def forward(self, pixel_values: torch.Tensor) -> VisionOutput:
        """
        Прямой проход через бэкбон.

        Args:
            pixel_values: Тензор [B, 3, H, W].

        Returns:
            VisionOutput с эмбеддингами и картами признаков.
        """
        # Глобальные признаки
        raw_embeddings = self.model(pixel_values)   # [B, backbone_dim]
        embeddings = self.proj(raw_embeddings)       # [B, feature_dim]

        # Патч-признаки (только ViT)
        patch_feats = (
            self._patch_features
            if self.is_vit and self._patch_features is not None
            else torch.zeros(pixel_values.size(0), 196, self.feature_dim,
                             device=pixel_values.device)
        )

        # Пространственные признаки (установлены хуком)
        spatial_feats = (
            self._spatial_features
            if self._spatial_features is not None
            else torch.zeros(pixel_values.size(0), self.feature_dim, 14, 14,
                             device=pixel_values.device)
        )

        # Опциональные логиты
        logits = self.classifier(embeddings) if self.classifier is not None else None

        return VisionOutput(
            embeddings=embeddings,
            patch_features=patch_feats,
            spatial_features=spatial_feats,
            logits=logits,
        )

    def freeze_backbone(self) -> None:
        """Заморозить веса бэкбона (только проекция и классификатор обучаются)."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, last_n_blocks: int = 2) -> None:
        """
        Разморозить последние N блоков бэкбона для тонкой настройки.

        Args:
            last_n_blocks: Количество размораживаемых блоков с конца.
        """
        # Сначала замораживаем всё
        for param in self.model.parameters():
            param.requires_grad = False

        # Размораживаем последние блоки
        if self.is_vit:
            blocks_to_unfreeze = self.model.blocks[-last_n_blocks:]
        else:
            all_blocks = list(self.model.children())
            blocks_to_unfreeze = all_blocks[-last_n_blocks:]

        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True

        # Проекцию и классификатор всегда обучаем
        for param in self.proj.parameters():
            param.requires_grad = True
        if self.classifier is not None:
            for param in self.classifier.parameters():
                param.requires_grad = True
