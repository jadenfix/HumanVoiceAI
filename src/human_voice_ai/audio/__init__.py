"""
Audio processing module for the Human Voice AI project.

This module provides audio feature extraction functionality including:
- Mel-spectrogram extraction
- Pitch (F0) estimation
- Energy calculation
"""

from .feature_extractor import FeatureExtractor

__all__ = ["FeatureExtractor"]
