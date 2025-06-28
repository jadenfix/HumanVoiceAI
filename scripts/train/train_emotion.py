#!/usr/bin/env python3
"""
Training script for the Speech Emotion Recognition (SER) model.
Implements the training pipeline with data augmentation, model training, and evaluation.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

from human_voice_ai.emotion.ser_model import SERModel
from human_voice_ai.emotion.ser_dataset import create_ser_data_loaders
from human_voice_ai.emotion.ser_trainer import SERTrainer
from human_voice_ai.utils.config import Config
from human_voice_ai.utils.metrics import plot_training_curves, save_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SER model')
    parser.add_argument('--config', type=str, default='configs/ser_config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='outputs/ser',
                        help='Directory to save outputs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda, mps, cpu)')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def setup_environment(seed: int = 42) -> None:
    """Set up the training environment.
    
    Args:
        seed: Random seed for reproducibility
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Log environment info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get the device to train on.
    
    Args:
        device_str: Device string (cuda, mps, cpu)
        
    Returns:
        torch.device: Device to use
    """
    if device_str is not None:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_model(config: Config, device: torch.device) -> nn.Module:
    """Create the SER model.
    
    Args:
        config: Configuration object
        device: Device to create the model on
        
    Returns:
        nn.Module: SER model
    """
    model = SERModel(config)
    model = model.to(device)
    
    # Log model summary
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Created model with {num_params:,} trainable parameters")
    
    return model


def create_optimizer(model: nn.Module, config: Config) -> optim.Optimizer:
    """Create the optimizer.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        optim.Optimizer: Optimizer
    """
    return optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        amsgrad=True
    )


def create_scheduler(optimizer: optim.Optimizer, config: Config) -> Any:
    """Create the learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        
    Returns:
        Learning rate scheduler or None
    """
    scheduler_config = config.training.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            verbose=True
        )
    elif scheduler_type == 'step':
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('t_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    else:
        return None


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment
    setup_environment(args.seed)
    
    # Set device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    config.training.num_epochs = args.num_epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.num_workers = args.num_workers
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(output_dir / 'config.yaml')
    
    # Create data loaders
    logger.info(f"Loading data from {args.data_dir}")
    data_loaders = create_ser_data_loaders(
        config=config,
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )
    
    # Create model
    logger.info("Creating model")
    model = create_model(config, device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = SERTrainer(
        config=config,
        model=model,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    logger.info("Starting training")
    trainer.train(config.training.num_epochs)
    
    # Test the model
    logger.info("Testing model")
    test_metrics = trainer.test()
    
    # Save final metrics
    metrics = {
        'train': trainer.metrics['train'],
        'val': trainer.metrics['val'],
        'test': test_metrics
    }
    save_metrics(metrics, output_dir / 'metrics.json')
    
    # Plot training curves
    plot_training_curves(
        train_metrics=metrics['train'],
        val_metrics=metrics['val'],
        output_file=output_dir / 'training_curves.png'
    )
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
