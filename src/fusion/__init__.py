"""Модуль слияния признаков и принятия решений."""

from .cross_attention import CrossAttentionFusion
from .decision_head import DecisionHead

__all__ = ["CrossAttentionFusion", "DecisionHead"]
