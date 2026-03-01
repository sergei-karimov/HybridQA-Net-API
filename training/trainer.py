"""
Тренер для fine-tuning HybridQA-Net.

Реализует:
- Цикл обучения с валидацией
- Планировщик learning rate (cosine с warmup)
- Сохранение лучших N чекпоинтов
- Метрики: accuracy, F1, AUC-ROC
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import QADataset, collate_fn
from src.pipeline import HybridQANet
from utils.helpers import ensure_dir
from utils.logger import setup_logger

logger = setup_logger("hybridqa.trainer")


class Trainer:
    """
    Тренер HybridQA-Net.

    Пример использования::

        trainer = Trainer(system, config)
        trainer.train(
            train_dataset=QADataset("data", split="train", augment=True),
            val_dataset=QADataset("data", split="val"),
        )
    """

    def __init__(
        self,
        system: HybridQANet,
        config: Optional[dict] = None,
    ):
        """
        Args:
            system: Экземпляр HybridQANet.
            config: Словарь конфигурации (секция training).
        """
        self.system = system
        self.cfg = config or system.config.get("training", {})
        self.device = system.device

        # Параметры обучения
        self.batch_size: int = self.cfg.get("batch_size", 16)
        self.lr: float = self.cfg.get("learning_rate", 1e-4)
        self.epochs: int = self.cfg.get("epochs", 50)
        self.warmup_steps: int = self.cfg.get("warmup_steps", 100)
        self.weight_decay: float = self.cfg.get("weight_decay", 0.01)
        self.gradient_clip: float = self.cfg.get("gradient_clip", 1.0)
        self.eval_every: int = self.cfg.get("eval_every", 5)
        self.save_top_k: int = self.cfg.get("save_top_k", 3)
        self.checkpoint_dir = ensure_dir(self.cfg.get("checkpoint_dir", "./checkpoints"))

        # Лучшие чекпоинты (min-heap по метрике)
        self._best_checkpoints: list[tuple[float, str]] = []

        # История обучения
        self.history: dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
        }

    # ================================================================= Train
    def train(
        self,
        train_dataset: QADataset,
        val_dataset: Optional[QADataset] = None,
        freeze_backbone: bool = True,
    ) -> dict:
        """
        Запустить цикл обучения.

        Args:
            train_dataset: Обучающий датасет.
            val_dataset: Валидационный датасет (опционально).
            freeze_backbone: Заморозить бэкбон.

        Returns:
            История обучения.
        """
        logger.info(f"Начало обучения: {self.epochs} эпох, batch={self.batch_size}")

        # Подготовка к fine-tuning
        self.system.fine_tune_prepare(freeze_backbone=freeze_backbone)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
            )
            if val_dataset
            else None
        )

        # Оптимизатор — AdamW с weight decay
        trainable_params = [p for p in self.system._all_params() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.epochs * len(train_loader)
        scheduler = self._build_scheduler(optimizer, total_steps)

        # Веса классов для несбалансированного датасета
        class_weights = train_dataset.class_weights.to(self.device)

        # Цикл обучения
        best_val_acc = 0.0
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            train_loss = self._train_epoch(
                train_loader, optimizer, scheduler, class_weights, global_step
            )
            global_step += len(train_loader)
            epoch_time = time.time() - epoch_start

            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(optimizer.param_groups[0]["lr"])

            logger.info(
                f"Эпоха {epoch}/{self.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Время: {epoch_time:.1f}с"
            )

            # Валидация
            if val_loader and epoch % self.eval_every == 0:
                val_loss, val_acc = self._evaluate(val_loader, class_weights)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)

                logger.info(
                    f"  Валидация → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}"
                )

                # Сохраняем лучшие чекпоинты
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self._save_best_checkpoint(epoch, val_acc)

        logger.info(f"Обучение завершено. Лучшая val accuracy: {best_val_acc:.4f}")
        return self.history

    # --------------------------------------------------------------- Epoch
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: AdamW,
        scheduler,
        class_weights: torch.Tensor,
        global_step: int,
    ) -> float:
        """Один проход обучения."""
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            queries = batch["queries"]
            documents = batch["documents"]

            # Forward pass
            vision_out = self.system.vision(pixel_values)
            context_out = self.system.context_analyzer(
                query=queries,
                document=documents,
            )
            fusion_out = self.system.fusion(
                vision_out.patch_features,
                context_out.token_embeddings,
            )
            decision_out = self.system.decision_head(fusion_out.fused_embedding)

            # Loss
            loss = self.system.decision_head.compute_loss(
                decision_out.logits,
                labels,
                class_weights=class_weights,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.system._all_params() if p.requires_grad],
                self.gradient_clip,
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # -------------------------------------------------------------- Eval
    @torch.no_grad()
    def _evaluate(
        self,
        loader: DataLoader,
        class_weights: torch.Tensor,
    ) -> tuple[float, float]:
        """Оценка на валидационной выборке."""
        self.system.vision.eval()
        self.system.context_analyzer.eval()
        self.system.fusion.eval()
        self.system.decision_head.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            queries = batch["queries"]
            documents = batch["documents"]

            vision_out = self.system.vision(pixel_values)
            context_out = self.system.context_analyzer(
                query=queries,
                document=documents,
            )
            fusion_out = self.system.fusion(
                vision_out.patch_features,
                context_out.token_embeddings,
            )
            decision_out = self.system.decision_head(fusion_out.fused_embedding)

            loss = self.system.decision_head.compute_loss(
                decision_out.logits, labels, class_weights
            )
            total_loss += loss.item()

            preds = decision_out.probabilities.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Возвращаем модули в train mode
        self.system.vision.train()
        self.system.context_analyzer.train()
        self.system.fusion.train()
        self.system.decision_head.train()

        val_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return val_loss, accuracy

    # --------------------------------------------------------------- Checkpoint
    def _save_best_checkpoint(self, epoch: int, metric: float) -> None:
        """Сохранить чекпоинт и удалить худший если >save_top_k."""
        path = str(self.checkpoint_dir / f"epoch_{epoch:03d}_acc{metric:.4f}.pt")
        self.system.save_checkpoint(path)

        # Добавляем в min-heap (метрика, путь)
        heapq.heappush(self._best_checkpoints, (metric, path))

        # Удаляем худший если превышен лимит
        if len(self._best_checkpoints) > self.save_top_k:
            worst_metric, worst_path = heapq.heappop(self._best_checkpoints)
            Path(worst_path).unlink(missing_ok=True)
            logger.debug(f"Удалён чекпоинт: {worst_path}")

    @staticmethod
    def _build_scheduler(optimizer: AdamW, total_steps: int):
        """CosineAnnealingLR планировщик."""
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
