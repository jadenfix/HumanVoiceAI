"""
Base model class for all neural network models in the VoiceAI system.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import torch
import torch.nn as nn

from human_voice_ai.utils.config import Config


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models.
    Provides common functionality for model initialization, saving, loading, and device management.
    """

    def __init__(self, config: Config, *args, **kwargs):
        """
        Initialize the base model.

        Args:
            config: Configuration object containing model hyperparameters
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.device = self._setup_device()
        self._build_model()
        self.to(self.device)

    @abstractmethod
    def _build_model(self) -> None:
        """
        Build the model architecture.
        This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the model.
        This method should be implemented by all subclasses.
        """
        pass

    def _setup_device(self) -> torch.device:
        """
        Set up the device for model training/inference.

        Returns:
            torch.device: The device to use for computation.
        """
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            return torch.device("cpu")

        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        else:
            device = torch.device(self.config.device)

        # Print device info
        if device.type == "cuda":
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif device.type == "mps":
            print("Using MPS (Apple Silicon) device")
        else:
            print("Using CPU (no GPU/accelerator found)")

        return device

    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        epoch: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            save_path: Path to save the checkpoint
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state to save
            epoch: Current epoch number
            **kwargs: Additional items to save in the checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "config": self.config.dict(),
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[Config] = None,
        device: str = "auto",
        **kwargs,
    ) -> Tuple["BaseModel", Dict[str, Any]]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            config: Optional config object (if None, will be loaded from checkpoint)
            device: Device to load the model onto
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            Tuple of (model, checkpoint) where checkpoint is a dictionary containing
            the loaded state and any additional saved items.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config (either from checkpoint or provided)
        if config is None:
            from human_voice_ai.utils.config import Config

            config = Config(**checkpoint["config"])

        # Override device in config if specified
        if device != "auto":
            config.device = device

        # Create model instance
        model = cls(config, **kwargs)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move model to appropriate device
        model.device = model._setup_device()
        model.to(model.device)

        print(f"Loaded checkpoint from {checkpoint_path}")

        # Remove model state dict to save memory
        checkpoint.pop("model_state_dict", None)

        return model, checkpoint

    def count_parameters(self) -> int:
        """
        Count the total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def summary(self) -> str:
        """
        Generate a summary of the model architecture.

        Returns:
            str: Model summary
        """
        from torchinfo import summary

        # Create a dummy input
        input_shape = self.get_input_shape()
        dummy_input = torch.randn(1, *input_shape).to(self.device)

        # Generate summary
        model_summary = summary(
            self,
            input_data=dummy_input,
            verbose=0,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "kernel_size",
                "mult_adds",
            ),
            depth=10,
        )

        return str(model_summary)

    def get_input_shape(self) -> tuple:
        """
        Get the expected input shape for the model.
        This should be implemented by subclasses.

        Returns:
            tuple: Expected input shape (excluding batch dimension)
        """
        raise NotImplementedError("Subclasses should implement get_input_shape()")
