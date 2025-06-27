"""
Text-to-Speech (TTS) engine for generating speech from text with emotion control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class TtsEngine(nn.Module):
    """Text-to-Speech engine with emotion control."""
    
    def __init__(self, 
                 num_chars: int = 100,
                 embedding_dim: int = 512,
                 encoder_dim: int = 512,
                 decoder_dim: int = 1024,
                 num_mels: int = 80,
                 dropout: float = 0.1):
        """Initialize the TTS engine.
        
        Args:
            num_chars: Number of characters in the vocabulary
            embedding_dim: Character embedding dimension
            encoder_dim: Encoder hidden dimension
            decoder_dim: Decoder hidden dimension
            num_mels: Number of mel filterbanks
            dropout: Dropout probability
        """
        super().__init__()
        self.num_mels = num_mels
        
        # Text encoder
        self.embedding = nn.Embedding(num_chars, embedding_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, encoder_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Emotion conditioning
        self.emotion_proj = nn.Linear(5, encoder_dim)  # 5 emotion classes
        
        # Decoder (autoregressive)
        self.prenet = nn.Sequential(
            nn.Linear(num_mels, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.lstm1 = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        self.lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)
        
        # Mel prediction
        self.mel_proj = nn.Linear(decoder_dim, num_mels)
        
        # Stop token prediction
        self.stop_proj = nn.Linear(decoder_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                elif 'proj' in name or 'embedding' in name:
                    if len(param.shape) >= 2:
                        nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)
    
    def forward(self, 
               text: torch.Tensor, 
               mel_target: Optional[torch.Tensor] = None,
               emotion: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training.
        
        Args:
            text: Input text tensor of shape [batch_size, seq_len]
            mel_target: Target mel spectrogram of shape [batch_size, mel_frames, num_mels]
            emotion: Emotion embedding of shape [batch_size, num_emotions]
            
        Returns:
            tuple: (mel_output, stop_tokens)
        """
        batch_size = text.size(0)
        
        # Encode text
        x = self.embedding(text).transpose(1, 2)  # [B, emb_dim, seq_len]
        encoder_output = self.encoder(x).transpose(1, 2)  # [B, seq_len, enc_dim]
        
        # Add emotion conditioning
        if emotion is not None:
            emotion_embedding = self.emotion_proj(emotion).unsqueeze(1)  # [B, 1, enc_dim]
            encoder_output = encoder_output + emotion_embedding
        
        # Initialize decoder states
        decoder_h1 = torch.zeros(batch_size, self.lstm1.hidden_size).to(text.device)
        decoder_c1 = torch.zeros_like(decoder_h1)
        decoder_h2 = torch.zeros(batch_size, self.lstm2.hidden_size).to(text.device)
        decoder_c2 = torch.zeros_like(decoder_h2)
        
        # Initialize output tensors
        max_frames = mel_target.size(1) if mel_target is not None else 1000
        mel_outputs = []
        stop_outputs = []
        
        # Initial input is zeros
        prev_mel = torch.zeros(batch_size, self.num_mels).to(text.device)
        
        # Teacher forcing loop
        for i in range(max_frames):
            # Prenet
            prenet_out = self.prenet(prev_mel)
            
            # Attention (simplified for this example)
            # In a full implementation, you would have an attention mechanism here
            context = encoder_output.mean(dim=1)  # Simple mean pooling
            
            # LSTM 1
            decoder_h1, decoder_c1 = self.lstm1(
                torch.cat([prenet_out, context], dim=1),
                (decoder_h1, decoder_c1)
            )
            
            # LSTM 2
            decoder_h2, decoder_c2 = self.lstm2(
                decoder_h1,
                (decoder_h2, decoder_c2)
            )
            
            # Predict mel and stop token
            mel_out = self.mel_proj(decoder_h2)
            stop_out = torch.sigmoid(self.stop_proj(decoder_h2))
            
            # Store outputs
            mel_outputs.append(mel_out.unsqueeze(1))
            stop_outputs.append(stop_out)
            
            # Teacher forcing
            if mel_target is not None and i < mel_target.size(1) - 1:
                prev_mel = mel_target[:, i, :]
            else:
                prev_mel = mel_out
                
            # Stop if all sequences have finished
            if torch.all(stop_out > 0.5):
                break
        
        # Stack outputs
        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_outputs = torch.cat(stop_outputs, dim=1)
        
        return mel_outputs, stop_outputs
    
    @torch.no_grad()
    def generate(self, 
                text: Union[str, torch.Tensor], 
                emotion: Optional[torch.Tensor] = None,
                max_frames: int = 1000,
                temperature: float = 0.667) -> torch.Tensor:
        """Generate mel spectrogram from text.
        
        Args:
            text: Input text or token tensor
            emotion: Emotion embedding of shape [num_emotions]
            max_frames: Maximum number of frames to generate
            temperature: Sampling temperature
            
        Returns:
            torch.Tensor: Generated mel spectrogram of shape [1, T, num_mels]
        """
        self.eval()
        
        # Tokenize text if needed
        if isinstance(text, str):
            # In a real implementation, you would use a proper tokenizer here
            text = torch.tensor([ord(c) for c in text], dtype=torch.long)
        
        text = text.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Process emotion
        if emotion is not None:
            if isinstance(emotion, (list, tuple, np.ndarray)):
                emotion = torch.tensor(emotion, dtype=torch.float32)
            emotion = emotion.unsqueeze(0).to(self.device)
        
        # Encode text
        x = self.embedding(text).transpose(1, 2)
        encoder_output = self.encoder(x).transpose(1, 2)
        
        # Add emotion conditioning
        if emotion is not None:
            emotion_embedding = self.emotion_proj(emotion).unsqueeze(1)
            encoder_output = encoder_output + emotion_embedding
        
        # Initialize decoder states
        batch_size = 1
        decoder_h1 = torch.zeros(batch_size, self.lstm1.hidden_size).to(self.device)
        decoder_c1 = torch.zeros_like(decoder_h1)
        decoder_h2 = torch.zeros(batch_size, self.lstm2.hidden_size).to(self.device)
        decoder_c2 = torch.zeros_like(decoder_h2)
        
        # Initialize output tensors
        mel_outputs = []
        prev_mel = torch.zeros(batch_size, self.num_mels).to(self.device)
        
        # Generation loop
        for i in range(max_frames):
            # Prenet
            prenet_out = self.prenet(prev_mel)
            
            # Attention (simplified)
            context = encoder_output.mean(dim=1)
            
            # LSTM 1
            decoder_h1, decoder_c1 = self.lstm1(
                torch.cat([prenet_out, context], dim=1),
                (decoder_h1, decoder_c1)
            )
            
            # LSTM 2
            decoder_h2, decoder_c2 = self.lstm2(
                decoder_h1,
                (decoder_h2, decoder_c2)
            )
            
            # Predict mel and stop token
            mel_out = self.mel_proj(decoder_h2)
            stop_out = torch.sigmoid(self.stop_proj(decoder_h2))
            
            # Store output
            mel_outputs.append(mel_out.unsqueeze(1))
            
            # Stop if finished
            if stop_out > 0.5:
                break
                
            # Update previous mel with temperature sampling
            if temperature > 0:
                mel_out = mel_out / temperature
                mel_out = torch.distributions.Normal(mel_out, 1.0).sample()
                
            prev_mel = mel_out
        
        # Stack outputs
        if not mel_outputs:
            return torch.zeros(1, 1, self.num_mels).to(self.device)
            
        mel_outputs = torch.cat(mel_outputs, dim=1)
        return mel_outputs
    
    def to(self, device):
        """Move model to device and store device."""
        self.device = device
        return super().to(device)

# For testing
if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = TtsEngine(num_chars=128).to(device)
    
    # Test forward pass
    text = torch.randint(0, 128, (16, 50)).to(device)
    mel = torch.randn(16, 100, 80).to(device)
    emotion = torch.randn(16, 5).to(device)
    
    out_mel, out_stop = model(text, mel, emotion)
    print(f"Output mel shape: {out_mel.shape}")
    print(f"Output stop shape: {out_stop.shape}")
    
    # Test generation
    test_text = "Hello, this is a test."
    generated_mel = model.generate(test_text, emotion[0])
    print(f"Generated mel shape: {generated_mel.shape}")
