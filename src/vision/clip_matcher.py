"""
Multilingual CLIP matcher using sentence-transformers.

Architecture:
- Image encoder: clip-ViT-B-32  (CLIP visual backbone)
- Text encoder:  clip-ViT-B-32-multilingual-v1  (multilingual, supports Russian)

Both models produce embeddings in the SAME semantic space, so cosine
similarity between image embeddings and multilingual text embeddings is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class RegionMatch:
    region_name: str
    similarity: float
    crop_width: int
    crop_height: int


@dataclass
class CLIPMatchResult:
    best_region: str
    best_similarity: float
    all_regions: list[RegionMatch]
    text_query: str


class CLIPMatcher:
    IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
    TEXT_MODEL  = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    def __init__(
        self,
        image_model_name: str = IMAGE_MODEL,
        text_model_name: str = TEXT_MODEL,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Two separate models: CLIP for images, multilingual distilled for text
        self.img_model = SentenceTransformer(image_model_name, device=self.device)
        self.txt_model = SentenceTransformer(text_model_name, device=self.device)

    def encode_images(self, crops: list[Image.Image]) -> torch.Tensor:
        """Encode PIL images via CLIP visual backbone → [N, D] normalized."""
        embs = self.img_model.encode(
            crops,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embs  # type: ignore[return-value]

    def encode_text(self, query: str) -> torch.Tensor:
        """Encode text via multilingual model → [D] normalized.
        Supports Russian natively (distilled to match CLIP image space).
        """
        emb = self.txt_model.encode(
            query,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return emb  # type: ignore[return-value]

    def match(
        self,
        crops: list[Image.Image],
        region_names: list[str],
        query: str,
    ) -> CLIPMatchResult:
        """
        Cosine similarity between each crop embedding and the text query.
        normalize_embeddings=True → dot product == cosine similarity.
        """
        img_embs = self.encode_images(crops)   # [N, D]
        txt_emb  = self.encode_text(query)     # [D]

        similarities = (img_embs @ txt_emb).cpu().tolist()

        regions: list[RegionMatch] = [
            RegionMatch(
                region_name=name,
                similarity=float(sim),
                crop_width=crop.width,
                crop_height=crop.height,
            )
            for name, sim, crop in zip(region_names, similarities, crops)
        ]

        best = max(regions, key=lambda r: r.similarity)

        return CLIPMatchResult(
            best_region=best.region_name,
            best_similarity=best.similarity,
            all_regions=regions,
            text_query=query,
        )
