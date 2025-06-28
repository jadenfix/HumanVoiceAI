"""
Configuration management system for the VoiceAI project.
Handles YAML configuration files with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type
import yaml
from pydantic import BaseModel, Field, validator

T = TypeVar("T", bound="ConfigBase")


class ConfigBase(BaseModel):
    """Base configuration class that supports YAML loading and environment overrides."""

    class Config:
        extra = "forbid"  # Forbid extra fields
        validate_assignment = True

    @classmethod
    def from_yaml(cls: Type[T], config_path: Path) -> T:
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.dict(exclude_none=True), f, default_flow_style=False)

    def update_from_env(self, prefix: str = "VOICE_AI_") -> None:
        """Update configuration from environment variables.

        Args:
            prefix: Prefix for environment variables to consider
        """
        for field_name, field in self.__fields__.items():
            env_name = f"{prefix}{field_name.upper()}"
            if env_name in os.environ:
                value = os.environ[env_name]
                # Convert string to appropriate type
                field_type = field.type_
                try:
                    if field_type == bool:
                        value = value.lower() in ("true", "1", "t", "y", "yes")
                    elif field_type == int:
                        value = int(value)
                    elif field_type == float:
                        value = float(value)
                    elif field_type == str:
                        pass  # Already a string
                    elif field_type == list:
                        value = [item.strip() for item in value.split(",")]
                    # Add more type conversions as needed
                except (ValueError, AttributeError) as e:
                    raise ValueError(
                        f"Failed to convert {env_name}={value} to {field_type}"
                    ) from e

                setattr(self, field_name, value)


class AudioConfig(ConfigBase):
    """Audio processing configuration."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: Optional[float] = None
    preemphasis: float = 0.97
    min_level_db: float = -100.0
    ref_level_db: float = 20.0
    max_wav_value: float = 32768.0

    @validator("fmax")
    def set_fmax(cls, v):
        return v or cls.sample_rate // 2


class SERConfig(ConfigBase):
    """Speech Emotion Recognition configuration."""

    model_type: str = "cnn_lstm"  # or 'transformer'
    num_classes: int = 5  # neutral, happy, sad, angry, surprised
    input_dim: int = 80  # Should match n_mels in AudioConfig
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 32
    num_epochs: int = 100
    use_attention: bool = True


class TTSConfig(ConfigBase):
    """Text-to-Speech configuration."""

    model_type: str = "tacotron2"  # or 'fastspeech2'
    vocab_size: int = 256
    embed_dim: int = 512
    encoder_dim: int = 256
    decoder_dim: int = 1024
    n_mel_channels: int = 80
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 16
    num_epochs: int = 1000
    use_gst: bool = True  # Global Style Tokens


class VocoderConfig(ConfigBase):
    """Vocoder configuration."""

    model_type: str = "hifigan"  # or 'waveglow'
    sample_rate: int = 22050
    n_fft: int = 1024
    num_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: Optional[float] = None

    # HiFi-GAN specific
    resblock: str = "1"  # '1' or '2'
    upsample_rates: list = [8, 8, 2, 2]
    upsample_kernel_sizes: list = [16, 16, 4, 4]
    upsample_initial_channel: int = 512
    resblock_kernel_sizes: list = [3, 7, 11]
    resblock_dilation_sizes: list = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    @validator("fmax")
    def set_fmax(cls, v):
        return v or 8000.0  # Default to 8kHz if not specified


class Config(ConfigBase):
    """Main configuration class that contains all sub-configurations."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    ser: SERConfig = Field(default_factory=SERConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vocoder: VocoderConfig = Field(default_factory=VocoderConfig)
    device: str = "auto"  # auto, cuda, mps, cpu
    debug: bool = False
    seed: int = 42

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Handle nested configurations
        for key in ["audio", "ser", "tts", "vocoder"]:
            if key in config_dict and isinstance(config_dict[key], dict):
                config_dict[key] = globals()[f"{key.capitalize()}Config"](
                    **config_dict[key]
                )

        return cls(**config_dict)

    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # Update main config
        super().update_from_env("VOICE_AI_")

        # Update sub-configs
        self.audio.update_from_env("VOICE_AI_AUDIO_")
        self.ser.update_from_env("VOICE_AI_SER_")
        self.tts.update_from_env("VOICE_AI_TTS_")
        self.vocoder.update_from_env("VOICE_AI_VOCODER_")

        # Handle device auto-detection
        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def save_config(config: Config, path: Path) -> None:
    """Save configuration to a YAML file."""
    config_dict = config.dict(exclude_none=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)


def load_config(path: Optional[Path] = None) -> Config:
    """Load configuration from a YAML file or return default."""
    if path is None:
        return get_default_config()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    return Config.from_yaml(path)
