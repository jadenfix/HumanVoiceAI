"""
Training and evaluation utilities for Speech Emotion Recognition (SER) models.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from human_voice_ai.base_trainer import BaseTrainer
from human_voice_ai.emotion.ser_model import SERModel
from human_voice_ai.emotion.ser_dataset import SERDataset
from human_voice_ai.utils.metrics import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_training_curves,
)


class SERTrainer(BaseTrainer):
    """
    Trainer class for Speech Emotion Recognition models.
    Handles training, validation, testing, and model checkpointing.
    """

    def __init__(
        self,
        config: "Config",
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize the SER trainer.

        Args:
            config: Configuration object
            model: SER model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            device: Device to train on
            **kwargs: Additional arguments for BaseTrainer
        """
        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            **kwargs,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.metrics = {
            "train": {"loss": [], "accuracy": []},
            "val": {"loss": [], "accuracy": []},
            "test": {"loss": [], "accuracy": []},
        }

        # Class weights for imbalanced datasets
        self.class_weights = self._compute_class_weights()
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def _compute_class_weights(self) -> Optional[torch.Tensor]:
        """
        Compute class weights based on training data distribution.

        Returns:
            Tensor of class weights or None if not using class weights
        """
        if (
            not hasattr(self.config.training, "use_class_weights")
            or not self.config.training.use_class_weights
        ):
            return None

        # Count samples per class
        class_counts = torch.zeros(len(SERDataset.EMOTION_MAP))
        for batch in self.train_loader:
            labels = batch["labels"]
            for label in labels:
                class_counts[label] += 1

        # Compute weights
        weights = 1.0 / (
            class_counts + 1e-6
        )  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum() * len(weights)  # Normalize

        return weights

    def _training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            batch: Batch of training data
            batch_idx: Batch index

        Returns:
            Dictionary containing loss and other metrics
        """
        # Move data to device
        features = batch["features"].to(self.device).unsqueeze(1)  # Add channel dim
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(features)
        logits = outputs["logits"]

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass and optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.grad_clip
            )

        self.optimizer.step()

        # Compute metrics
        _, preds = torch.max(logits, 1)
        acc = (preds == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": acc,
            "predictions": preds.detach().cpu(),
            "targets": labels.detach().cpu(),
        }

    @torch.no_grad()
    def _validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        Args:
            batch: Batch of validation data
            batch_idx: Batch index

        Returns:
            Dictionary containing loss and other metrics
        """
        # Move data to device
        features = batch["features"].to(self.device).unsqueeze(1)  # Add channel dim
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(features)
        logits = outputs["logits"]

        # Compute loss
        loss = self.criterion(logits, labels)

        # Compute metrics
        _, preds = torch.max(logits, 1)
        acc = (preds == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": acc,
            "predictions": preds.cpu(),
            "targets": labels.cpu(),
            "attention_weights": outputs.get("attention_weights", None),
        }

    def _test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single test step.

        Args:
            batch: Batch of test data
            batch_idx: Batch index

        Returns:
            Dictionary containing loss and other metrics
        """
        # Same as validation step but with additional metrics
        metrics = self._validation_step(batch, batch_idx)

        # Add filepaths for per-sample analysis
        metrics["filepaths"] = batch.get("filepaths", [])

        return metrics

    def _log_metrics(
        self,
        split: str,
        metrics: Dict[str, torch.Tensor],
        global_step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to TensorBoard and console.

        Args:
            split: Data split ('train', 'val', 'test')
            metrics: Dictionary of metrics to log
            global_step: Global step for logging
        """
        if global_step is None:
            global_step = self.global_step

        # Log to TensorBoard
        if self.use_tensorboard:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, torch.Tensor)):
                    self.writer.add_scalar(
                        f"{split}/{metric_name}", metric_value, global_step
                    )

        # Update metrics history
        for metric_name, metric_value in metrics.items():
            if metric_name in self.metrics[split]:
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.item()
                self.metrics[split][metric_name].append(metric_value)

    def _save_checkpoint(self, val_loss: float) -> None:
        """
        Save model checkpoint if validation loss improved.

        Args:
            val_loss: Current validation loss
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, "latest.pth")
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "val_loss": val_loss,
                "metrics": self.metrics,
                "config": self.config.dict(),
            },
            checkpoint_path,
        )

        # Save best checkpoint if validation loss improved
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.pth")
            torch.save(
                {
                    "epoch": self.epoch,
                    "global_step": self.global_step,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict()
                    if self.scheduler
                    else None,
                    "val_loss": val_loss,
                    "metrics": self.metrics,
                    "config": self.config.dict(),
                },
                best_checkpoint_path,
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if available
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if (
            "scheduler_state_dict" in checkpoint
            and self.scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load training state
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("val_loss", float("inf"))
        self.metrics = checkpoint.get(
            "metrics",
            {
                "train": {"loss": [], "accuracy": []},
                "val": {"loss": [], "accuracy": []},
                "test": {"loss": [], "accuracy": []},
            },
        )

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, global step {self.global_step}")
        print(f"Best validation loss: {self.best_metric:.4f}")

    def evaluate(
        self, data_loader: DataLoader, split: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate the model on a given dataset.

        Args:
            data_loader: DataLoader for evaluation data
            split: Data split name for logging

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(data_loader, desc=f"Evaluating {split}")
            ):
                # Move data to device
                features = (
                    batch["features"].to(self.device).unsqueeze(1)
                )  # Add channel dim
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(features)
                logits = outputs["logits"]

                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * features.size(0)

                # Get predictions
                _, preds = torch.max(logits, 1)

                all_predictions.append(preds.cpu())
                all_targets.append(labels.cpu())

        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metrics
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = (all_predictions == all_targets).float().mean().item()

        # Calculate additional metrics
        metrics = calculate_metrics(
            predictions=all_predictions.numpy(),
            targets=all_targets.numpy(),
            class_names=list(SERDataset.EMOTION_MAP.keys()),
        )

        # Log metrics
        metrics.update({f"{split}/loss": avg_loss, f"{split}/accuracy": accuracy})

        # Plot confusion matrix
        if hasattr(self, "writer") and self.writer is not None:
            fig = plot_confusion_matrix(
                y_true=all_targets.numpy(),
                y_pred=all_predictions.numpy(),
                class_names=list(SERDataset.EMOTION_MAP.keys()),
            )
            self.writer.add_figure(f"{split}/confusion_matrix", fig, self.global_step)

        return metrics


def train_ser_model(
    config: "Config",
    data_dir: str,
    output_dir: str,
    device: Optional[torch.device] = None,
    resume_from: Optional[str] = None,
) -> Tuple[SERModel, Dict[str, float]]:
    """
    Train a Speech Emotion Recognition model.

    Args:
        config: Configuration object
        data_dir: Directory containing training data
        output_dir: Directory to save model checkpoints and logs
        device: Device to train on
        resume_from: Path to checkpoint to resume training from

    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.dict(), f, indent=2)

    # Create data loaders
    data_loaders = SERDataset.create_data_loaders(
        config=config,
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    # Create model
    model = SERModel(config).to(device)

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create learning rate scheduler
    if config.training.scheduler == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
    elif config.training.scheduler == "step":
        scheduler = StepLR(
            optimizer, step_size=config.training.step_size, gamma=config.training.gamma
        )
    elif config.training.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # Create trainer
    trainer = SERTrainer(
        config=config,
        model=model,
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        test_loader=data_loaders["test"],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
    )

    # Load checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        trainer.load_checkpoint(resume_from)

    # Train the model
    trainer.train(config.training.num_epochs)

    # Evaluate on test set
    test_metrics = trainer.evaluate(data_loaders["test"], split="test")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.dict(),
            "test_metrics": test_metrics,
        },
        os.path.join(output_dir, "final_model.pth"),
    )

    return model, test_metrics
