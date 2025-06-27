"""
Base trainer class for training and evaluating models.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from human_voice_ai.utils.config import Config


class BaseTrainer:
    """
    Base class for all trainers.
    Handles training loop, validation, testing, and model checkpointing.
    """
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
        output_dir: str = 'outputs',
        use_tensorboard: bool = True,
        **kwargs
    ):
        """
        Initialize the base trainer.
        
        Args:
            config: Configuration object
            model: Model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save outputs
            use_tensorboard: Whether to use TensorBoard for logging
            **kwargs: Additional arguments
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_tensorboard = use_tensorboard
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.start_epoch = 0
        
        # Setup output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup TensorBoard
        self.writer = None
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
    
    def train(self, num_epochs: int) -> None:
        """
        Train the model for the specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
        """
        for epoch in range(self.start_epoch, num_epochs):
            self.epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch()
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
                
                # Update learning rate scheduler based on validation loss
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get('val_loss', 0))
                    else:
                        self.scheduler.step()
            
            # Log metrics
            self._log_metrics('train', train_metrics)
            if val_metrics:
                self._log_metrics('val', val_metrics)
            
            # Save checkpoint
            if val_metrics and 'val_loss' in val_metrics:
                self._save_checkpoint(val_metrics['val_loss'])
            
            # Print progress
            self._print_progress(epoch, num_epochs, train_metrics, val_metrics)
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Training logic for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}", unit="batch") as t:
            for batch_idx, batch in enumerate(t):
                # Training step
                step_metrics = self._training_step(batch, batch_idx)
                
                # Update metrics
                for k, v in step_metrics.items():
                    if k in epoch_metrics and isinstance(v, (int, float, torch.Tensor)):
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        epoch_metrics[k] += v
                
                # Update progress bar
                t.set_postfix({
                    'loss': step_metrics.get('loss', 0).item() if hasattr(step_metrics.get('loss', 0), 'item') else step_metrics.get('loss', 0),
                    'acc': step_metrics.get('accuracy', 0).item() if hasattr(step_metrics.get('accuracy', 0), 'item') else step_metrics.get('accuracy', 0)
                })
                
                # Increment global step
                self.global_step += 1
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validation logic for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        epoch_metrics = {
            'val_loss': 0.0,
            'val_accuracy': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Validation step
                step_metrics = self._validation_step(batch, batch_idx)
                
                # Update metrics
                for k, v in step_metrics.items():
                    if k in epoch_metrics and isinstance(v, (int, float, torch.Tensor)):
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        epoch_metrics[k] += v
        
        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        return epoch_metrics
    
    def test(self) -> Dict[str, float]:
        """
        Test the model.
        
        Returns:
            Dictionary of test metrics
        """
        if self.test_loader is None:
            raise ValueError("Test loader not provided")
        
        self.model.eval()
        
        test_metrics = {
            'test_loss': 0.0,
            'test_accuracy': 0.0
        }
        
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Test step
                step_metrics = self._test_step(batch, batch_idx)
                
                # Update metrics
                for k, v in step_metrics.items():
                    if k in test_metrics and isinstance(v, (int, float, torch.Tensor)):
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        test_metrics[k] += v
        
        # Average metrics
        for k in test_metrics:
            test_metrics[k] /= num_batches
        
        # Log test metrics
        self._log_metrics('test', test_metrics)
        
        return test_metrics
    
    def _training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Batch index
            
        Returns:
            Dictionary of metrics for the step
        """
        raise NotImplementedError("Training step must be implemented by subclass")
    
    def _validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.
        
        Args:
            batch: Batch of validation data
            batch_idx: Batch index
            
        Returns:
            Dictionary of metrics for the step
        """
        raise NotImplementedError("Validation step must be implemented by subclass")
    
    def _test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single test step.
        
        Args:
            batch: Batch of test data
            batch_idx: Batch index
            
        Returns:
            Dictionary of metrics for the step
        """
        # By default, use validation step for testing
        return self._validation_step(batch, batch_idx)
    
    def _log_metrics(
        self,
        split: str,
        metrics: Dict[str, float],
        global_step: Optional[int] = None
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
        if self.use_tensorboard and self.writer is not None:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, torch.Tensor)):
                    self.writer.add_scalar(f"{split}/{metric_name}", metric_value, global_step)
    
    def _save_checkpoint(self, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            val_loss: Current validation loss
        """
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'val_loss': val_loss,
            'best_metric': self.best_metric,
            'config': self.config.dict() if hasattr(self.config, 'dict') else {}
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if val_loss < self.best_metric:
            self.best_metric = val_loss
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.start_epoch = self.epoch + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch}, global step {self.global_step})")
    
    def _print_progress(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """
        Print training progress.
        
        Args:
            epoch: Current epoch
            num_epochs: Total number of epochs
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Format metrics
        train_loss = train_metrics.get('loss', 0)
        train_acc = train_metrics.get('accuracy', 0)
        
        val_loss = val_metrics.get('val_loss', 'N/A')
        val_acc = val_metrics.get('val_accuracy', 'N/A')
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if val_metrics:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def __del__(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
