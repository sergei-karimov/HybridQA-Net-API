"""
Голова принятия решений.

Выполняет бинарную классификацию (соответствует / не соответствует стандарту)
и вычисляет оценку уверенности (confidence score).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecisionOutput:
    """Выход головы принятия решений."""

    label: int                      # 0 = не соответствует, 1 = соответствует
    confidence: float               # Уверенность [0, 1]
    logits: torch.Tensor            # [B, num_classes] — сырые логиты
    probabilities: torch.Tensor     # [B, num_classes] — вероятности после softmax
    defect_score: float             # Оценка вероятности дефекта [0, 1]


class DecisionHead(nn.Module):
    """
    Классификационная голова с дополнительной оценкой дефектов.

    Архитектура:
    - Глубокая сеть с остаточными связями
    - Dropout для регуляризации
    - Температурное масштабирование для калибровки уверенности
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Размерность входного эмбеддинга.
            hidden_dim: Размерность скрытого слоя.
            num_classes: Число классов (2 для бинарной классификации).
            dropout: Коэффициент dropout.
            temperature: Температура для масштабирования логитов.
        """
        super().__init__()
        self.num_classes = num_classes

        # Основной классификатор
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Голова оценки уверенности (calibration head)
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Температурный параметр (learnable)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        fused_embedding: torch.Tensor,   # [B, D]
    ) -> DecisionOutput:
        """
        Классифицировать по слитому вектору признаков.

        Args:
            fused_embedding: Вектор признаков [B, D].

        Returns:
            DecisionOutput с меткой, уверенностью и вероятностями.
        """
        # Логиты с температурным масштабированием
        logits = self.classifier(fused_embedding)                      # [B, C]
        scaled_logits = logits / self.temperature.clamp(min=0.1)
        probabilities = F.softmax(scaled_logits, dim=-1)               # [B, C]

        # Уверенность модели
        confidence_score = self.confidence_head(fused_embedding)       # [B, 1]

        # Финальное предсказание (argmax)
        predicted_label = probabilities.argmax(dim=-1)                 # [B]

        # Скалярные значения для одного примера
        label_val = predicted_label[0].item()
        conf_val = confidence_score[0, 0].item()

        # Оценка дефекта = вероятность класса "не соответствует" (класс 0)
        defect_score = probabilities[0, 0].item()

        return DecisionOutput(
            label=int(label_val),
            confidence=float(conf_val),
            logits=logits,
            probabilities=probabilities,
            defect_score=float(defect_score),
        )

    def compute_loss(
        self,
        logits: torch.Tensor,            # [B, C]
        targets: torch.Tensor,           # [B]
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Вычислить взвешенный cross-entropy loss.

        Args:
            logits: Логиты [B, C].
            targets: Метки классов [B].
            class_weights: Веса классов для несбалансированных датасетов.

        Returns:
            Скалярный loss.
        """
        return F.cross_entropy(logits, targets, weight=class_weights)
