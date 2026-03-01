"""
Модуль семантического анализа текста.

Использует RuBERT (DeepPavlov/rubert-base-cased) или LaBSE для:
- Кодирования текстовых запросов
- Кодирования стандартов качества
- Формирования контекстных векторов из истории диалога
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class ContextOutput:
    """Выход контекстного анализатора."""

    query_embedding: torch.Tensor        # [B, D] — эмбеддинг запроса
    doc_embedding: torch.Tensor          # [B, D] — эмбеддинг документа
    context_vector: torch.Tensor         # [B, D] — объединённый контекстный вектор
    token_embeddings: torch.Tensor       # [B, L, D] — поэлементные эмбеддинги запроса


class ContextAnalyzer(nn.Module):
    """
    Анализатор семантического контекста на базе трансформера.

    Поддерживает:
    - ``DeepPavlov/rubert-base-cased`` — RuBERT для русского языка
    - ``sentence-transformers/LaBSE`` — многоязычная модель
    """

    def __init__(
        self,
        model_name: str = "DeepPavlov/rubert-base-cased",
        max_length: int = 512,
        feature_dim: int = 768,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model_name: HuggingFace имя модели.
            max_length: Максимальная длина токенов.
            feature_dim: Выходная размерность эмбеддинга.
            device: Устройство для вычислений.
        """
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.device = device or torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Размерность скрытого слоя трансформера
        hidden_dim = self.encoder.config.hidden_size

        # Проекция в нужную размерность
        self.proj = (
            nn.Linear(hidden_dim, feature_dim)
            if hidden_dim != feature_dim
            else nn.Identity()
        )

        # Слой слияния: конкатенируем запрос + документ → контекстный вектор
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.feature_dim = feature_dim

    def encode(
        self,
        texts: list[str],
        pool: str = "cls",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Закодировать список текстов в эмбеддинги.

        Args:
            texts: Список строк.
            pool: Стратегия пулинга — "cls" или "mean".

        Returns:
            Кортеж (pooled_embedding [B, D], token_embeddings [B, L, D]).
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(**encoded)

        token_embs = outputs.last_hidden_state   # [B, L, D]

        if pool == "cls":
            pooled = token_embs[:, 0, :]         # CLS-токен
        else:
            # Mean pooling с учётом маски
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)

        pooled = self.proj(pooled)               # [B, feature_dim]
        return pooled, token_embs

    def forward(
        self,
        query: list[str],
        document: list[str],
        history: Optional[list[str]] = None,
    ) -> ContextOutput:
        """
        Полный проход: анализ запроса + документа + истории диалога.

        Args:
            query: Список запросов (батч).
            document: Список документов стандарта.
            history: Опциональная история диалога.

        Returns:
            ContextOutput с эмбеддингами и контекстным вектором.
        """
        # Если есть история — добавить её к запросу
        if history:
            combined_queries = [
                f"{' '.join(history)} [SEP] {q}" for q in query
            ]
        else:
            combined_queries = query

        # Кодируем запросы и документы параллельно
        query_emb, query_tokens = self.encode(combined_queries, pool="cls")
        doc_emb, _ = self.encode(document, pool="mean")

        # Слияние запрос + документ
        context = self.fusion(
            torch.cat([query_emb, doc_emb], dim=-1)
        )                                         # [B, feature_dim]

        return ContextOutput(
            query_embedding=query_emb,
            doc_embedding=doc_emb,
            context_vector=context,
            token_embeddings=query_tokens,
        )

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Кодировать большой список текстов батчами.

        Args:
            texts: Полный список текстов.
            batch_size: Размер мини-батча.

        Returns:
            Тензор эмбеддингов [N, D].
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb, _ = self.encode(batch)
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)

    def freeze(self) -> None:
        """Заморозить веса трансформера."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_last_layers(self, n_layers: int = 2) -> None:
        """Разморозить последние N слоёв трансформера."""
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Разморозить последние N энкодер-блоков
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for layer in self.encoder.encoder.layer[-n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Проекция и fusion всегда обучаются
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.fusion.parameters():
            param.requires_grad = True
