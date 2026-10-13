"""
Speech Emotion Recognition (SER) model implementation.
Uses a CNN-LSTM architecture with attention for emotion classification from audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from human_voice_ai.base_model import BaseModel
from human_voice_ai.utils.config import Config


class SERModel(BaseModel):
    """
    Speech Emotion Recognition model using CNN-LSTM architecture with attention.
    Takes mel-spectrograms as input and predicts emotion probabilities.
    """

    def __init__(self, config: Config):
        """
        Initialize the SER model.

        Args:
            config: Configuration object containing model hyperparameters
        """
        super().__init__(config)
        self.ser_config = config.ser
        if self.ser_config.model_type != "cnn_lstm":
            raise ValueError("Only cnn_lstm SER model_type is implemented")
        if config.audio.n_mels < 8 or self.ser_config.input_dim != config.audio.n_mels:
            raise ValueError("SER input_dim must match audio.n_mels >=8")

        # Input feature dimensions
        self.n_mels = config.audio.n_mels
        self.n_classes = self.ser_config.num_classes

        # Build model components
        self._build_model()

        # Initialize weights
        self._init_weights()
        self.to(self.device)

    def _build_model(self) -> None:
        """Build the model architecture."""
        # CNN feature extractor
        self.conv_layers = nn.Sequential(
            # Input shape: (batch, 1, n_mels, time)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),
        )

        # Calculate CNN output dimensions
        # This will be used to set up the LSTM input size
        self.conv_output_dim = self._get_conv_output_dim()

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.conv_output_dim,
            hidden_size=self.ser_config.hidden_dim,
            num_layers=self.ser_config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.ser_config.dropout if self.ser_config.num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.ser_config.hidden_dim * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.ser_config.dropout),
            nn.Linear(self.ser_config.hidden_dim * 2, self.ser_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.ser_config.dropout),
            nn.Linear(self.ser_config.hidden_dim, self.n_classes),
        )

    def _get_conv_output_dim(self) -> int:
        """Calculate the output dimension of the CNN feature extractor."""
        # Three2x2 pools; no dummy forward that mutates BatchNorm statistics.
        return 256 * (self.n_mels // 8)

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SER model.

        Args:
            x: Input tensor of shape (batch, channels, n_mels, time)

        Returns:
            Dictionary containing:
                - logits: Raw model outputs (batch, n_classes)
                - probs: Class probabilities (batch, n_classes)
                - attention_weights: Attention weights (batch, time)
        """
        if x.ndim != 4 or x.size(1) != 1 or x.size(2) != self.n_mels or x.size(3) < 8:
            raise ValueError("Expected [batch,1,n_mels,time>=8] mel input")
        if not torch.isfinite(x).all():
            raise ValueError("SER input must be finite")
        batch_size = x.size(0)

        # CNN feature extraction
        # Input shape: (batch, 1, n_mels, time)
        x = self.conv_layers(x)

        # Reshape for LSTM: (batch, time, features)
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, height)
        x = x.reshape(batch_size, x.size(1), -1)  # (batch, time, channels * height)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim * 2)

        # Attention mechanism
        if self.ser_config.use_attention:
            attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        else:
            attention_weights = torch.full_like(lstm_out[:, :, :1], 1 / lstm_out.size(1))

        # Apply attention weights
        context_vector = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, hidden_dim * 2)

        return context_vector, attention_weights.squeeze(-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        context_vector, attention_weights = self._encode(x)
        logits = self.classifier(context_vector)  # (batch, n_classes)

        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=1),
            "attention_weights": attention_weights,
            "context_vector": context_vector,
        }

    def get_input_shape(self) -> tuple:
        """Get the expected input shape for the model."""
        return (1, self.n_mels, 100)  # (channels, height, width)

    @classmethod
    def from_config(cls, config: Config) -> "SERModel":
        """
        Create an SERModel instance from a config object.

        Args:
            config: Configuration object

        Returns:
            SERModel instance
        """
        return cls(config)

    def predict_emotion(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion probabilities for input features.

        Args:
            x: Input tensor of shape (batch, channels, n_mels, time)

        Returns:
            Tuple of (probabilities, predicted_class_indices)
        """
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            probs = outputs["probs"]
            preds = torch.argmax(probs, dim=1)
            return probs, preds


class SERModelWithFeatures(SERModel):
    """
    Extended SER model that can take additional handcrafted features.
    Combines CNN-LSTM features with handcrafted acoustic features.
    """

    def __init__(self, config: Config, feature_dim: int = 0):
        """
        Initialize the SER model with additional features.

        Args:
            config: Configuration object
            feature_dim: Dimension of additional handcrafted features
        """
        self.feature_dim = feature_dim
        super().__init__(config)

    def _build_model(self) -> None:
        """Build the model architecture with additional feature support."""
        # First build the base model
        super()._build_model()

        if self.feature_dim > 0:
            # Add an additional layer to process handcrafted features
            self.feature_processor = nn.Sequential(
                nn.Linear(self.feature_dim, 64),
                nn.ReLU(),
                nn.Dropout(self.ser_config.dropout),
            )

            # Update classifier to combine CNN-LSTM and handcrafted features
            self.classifier = nn.Sequential(
                nn.Dropout(self.ser_config.dropout),
                nn.Linear(self.ser_config.hidden_dim * 2 + 64, self.ser_config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.ser_config.dropout),
                nn.Linear(self.ser_config.hidden_dim, self.n_classes),
            )

    def forward(
        self, x: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional handcrafted features.

        Args:
            x: Input tensor of shape (batch, channels, n_mels, time)
            features: Optional tensor of handcrafted features (batch, feature_dim)

        Returns:
            Dictionary containing model outputs
        """
        context, attention = self._encode(x)
        if self.feature_dim > 0:
            if features is None or features.shape != (x.size(0), self.feature_dim):
                raise ValueError("Configured handcrafted features require [batch,feature_dim] input")
            if not torch.isfinite(features).all():
                raise ValueError("Handcrafted features must be finite")
            context = torch.cat([context, self.feature_processor(features)], dim=1)
        logits = self.classifier(context)
        return {"logits": logits, "probs": F.softmax(logits, dim=1),
                "attention_weights": attention, "context_vector": context}



def create_ser_model(config: Config, feature_dim: int = 0) -> SERModel:
    """
    Factory function to create an SER model based on configuration.

    Args:
        config: Configuration object
        feature_dim: Dimension of additional handcrafted features (if any)

    Returns:
        Initialized SER model
    """
    if feature_dim > 0:
        return SERModelWithFeatures(config, feature_dim)
    return SERModel(config)
