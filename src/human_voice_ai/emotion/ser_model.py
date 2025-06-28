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
        self.config = config.ser

        # Input feature dimensions
        self.n_mels = config.audio.n_mels
        self.n_classes = self.config.num_classes

        # Build model components
        self._build_model()

        # Initialize weights
        self._init_weights()

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
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.n_classes),
        )

    def _get_conv_output_dim(self) -> int:
        """Calculate the output dimension of the CNN feature extractor."""
        # Create a dummy input to calculate the CNN output size
        dummy_input = torch.zeros(
            1, 1, self.n_mels, 100
        )  # (batch, channels, n_mels, time)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return output.size(1) * output.size(2)  # channels * height

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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        attention_weights = self.attention(lstm_out)  # (batch, time, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights
        context_vector = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, hidden_dim * 2)

        # Classifier
        logits = self.classifier(context_vector)  # (batch, n_classes)

        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=1),
            "attention_weights": attention_weights.squeeze(-1),  # (batch, time)
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
                nn.Dropout(self.config.dropout),
            )

            # Update classifier to combine CNN-LSTM and handcrafted features
            self.classifier = nn.Sequential(
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim * 2 + 64, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.n_classes),
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
        # Get base model outputs
        outputs = super().forward(x)

        # Process handcrafted features if provided
        if hasattr(self, "feature_processor") and features is not None:
            processed_features = self.feature_processor(features)

            # Combine CNN-LSTM features with handcrafted features
            combined = torch.cat([outputs["context_vector"], processed_features], dim=1)

            # Final classification
            logits = self.classifier(combined)
            outputs.update({"logits": logits, "probs": F.softmax(logits, dim=1)})

        return outputs


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
