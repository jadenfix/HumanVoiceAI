"""
Speech Emotion Recognition (SER) model for detecting emotions from audio features.
"""

import torch
import torch.nn as nn


class SerModel(nn.Module):
    """Speech Emotion Recognition model for detecting emotions from audio features."""

    def __init__(
        self, num_emotions=5, input_dim=82, hidden_dim=256, num_layers=2, dropout=0.2
    ):
        """Initialize the SER model.

        Args:
            num_emotions: Number of emotion classes
            input_dim: Input feature dimension (n_mels + pitch + energy)
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                elif "classifier" in name:
                    if len(param.shape) >= 2:
                        nn.init.xavier_normal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0.0)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Emotion logits of shape [batch_size, num_emotions]
        """
        # LSTM layers
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim * 2]

        # Take the last time step's output
        last_out = lstm_out[:, -1, :]  # [batch_size, hidden_dim * 2]

        # Classifier
        logits = self.classifier(last_out)  # [batch_size, num_emotions]

        return logits

    def predict_emotion(self, features):
        """Predict emotion from audio features.

        Args:
            features: Input features of shape [seq_len, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            tuple: (emotion_idx, emotion_probs)
        """
        self.eval()
        with torch.no_grad():
            if len(features.shape) == 2:
                features = features.unsqueeze(0)  # Add batch dimension

            # Move to device if not already
            if hasattr(self, "device"):
                features = features.to(self.device)

            # Forward pass
            logits = self(features)
            probs = torch.softmax(logits, dim=-1)

            # Get predicted class
            _, preds = torch.max(probs, dim=1)

            return preds.cpu().numpy(), probs.cpu().numpy()

    def to(self, device):
        """Move model to device and store device."""
        self.device = device
        return super().to(device)
