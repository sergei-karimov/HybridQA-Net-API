"""
Cross-Attention механизм для слияния визуальных и текстовых признаков.

Реализует двунаправленный cross-attention:
- Изображение → Текст (визуальные признаки обращаются к текстовому контексту)
- Текст → Изображение (текст обращается к визуальным признакам)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionOutput:
    """Выход модуля слияния признаков."""

    fused_embedding: torch.Tensor       # [B, D] — объединённый вектор
    visual_attended: torch.Tensor       # [B, D] — визуальный вектор после cross-attention
    text_attended: torch.Tensor         # [B, D] — текстовый вектор после cross-attention
    attention_weights: torch.Tensor     # [B, H, N, M] — веса внимания


class CrossAttentionLayer(nn.Module):
    """
    Один слой cross-attention: query из одной модальности, key/value из другой.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,   # [B, Nq, D]
        context: torch.Tensor, # [B, Nc, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output [B, Nq, D], attention_weights [B, Nq, Nc])
        """
        # Cross-attention: query → context
        attended, weights = self.attn(query, context, context)
        query = self.norm1(query + attended)

        # Feed-forward
        query = self.norm2(query + self.ffn(query))

        return query, weights


class CrossAttentionFusion(nn.Module):
    """
    Многослойный двунаправленный cross-attention для слияния модальностей.

    Схема работы:
    1. Проекция визуальных и текстовых признаков в единое пространство.
    2. Применение N слоёв cross-attention в обоих направлениях.
    3. Объединение результатов и нормализация.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            embed_dim: Размерность эмбеддинга.
            num_heads: Число голов внимания.
            num_layers: Число слоёв cross-attention.
            dropout: Dropout.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Visual: изображение обращается к тексту
        self.visual_to_text_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Text: текст обращается к изображению
        self.text_to_visual_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Финальное слияние двух векторов
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_features: torch.Tensor,   # [B, Nv, D] или [B, D]
        text_features: torch.Tensor,     # [B, Nt, D] или [B, D]
    ) -> FusionOutput:
        """
        Применить cross-attention fusion.

        Args:
            visual_features: Визуальные признаки (патч-токены или глобал).
            text_features: Текстовые признаки (токены или глобал).

        Returns:
            FusionOutput с объединёнными признаками.
        """
        # Нормализация формы: [B, D] → [B, 1, D]
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)

        v = visual_features   # [B, Nv, D]
        t = text_features     # [B, Nt, D]

        last_attn_vt = None
        last_attn_tv = None

        # Применяем слои cross-attention
        for v2t_layer, t2v_layer in zip(
            self.visual_to_text_layers,
            self.text_to_visual_layers,
        ):
            v, last_attn_vt = v2t_layer(v, t)  # visual обращается к тексту
            t, last_attn_tv = t2v_layer(t, v)  # текст обращается к визуалу

        # Пулинг: среднее по последовательности
        v_pooled = v.mean(dim=1)   # [B, D]
        t_pooled = t.mean(dim=1)   # [B, D]

        # Финальное слияние
        fused = self.final_fusion(torch.cat([v_pooled, t_pooled], dim=-1))  # [B, D]

        return FusionOutput(
            fused_embedding=fused,
            visual_attended=v_pooled,
            text_attended=t_pooled,
            attention_weights=last_attn_vt if last_attn_vt is not None else torch.zeros(1),
        )
