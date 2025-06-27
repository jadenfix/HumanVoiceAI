"""
Main demo script for the Human Voice AI project.
This script demonstrates the full pipeline from text to speech with emotion control.
"""

import argparse
import time
import os
import numpy as np
import torch
import torchaudio
import sounddevice as sd
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.absolute()))

from src.audio.feature_extractor import FeatureExtractor
from src.emotion.ser_model import SerModel
from src.tts.tts_engine import TtsEngine
from src.vocoder.hifigan import HiFiGAN

# Constants
SAMPLE_RATE = 22050
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

class HumanVoiceAI:
    """Main class for the Human Voice AI demo."""
    
    def __init__(self, device=DEVICE):
        """Initialize the demo with all required components."""
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize components
        print("Initializing components...")
        self.feature_extractor = FeatureExtractor(sample_rate=SAMPLE_RATE).to(self.device)
        self.ser_model = SerModel().to(self.device)
        self.tts_engine = TtsEngine().to(self.device)
        self.vocoder = HiFiGAN().to(self.device)
        
        # Load pretrained weights (in a real implementation, you would load actual weights)
        self._load_pretrained_weights()
        
        print("Initialization complete!")
    
    def _load_pretrained_weights(self):
        """Load pretrained weights for all models."""
        print("Loading pretrained weights...")
        
        # In a real implementation, you would load actual pretrained weights here
        # For example:
        # checkpoint = torch.load('path/to/checkpoint.pth', map_location=self.device)
        # self.ser_model.load_state_dict(checkpoint['ser_model'])
        # self.tts_engine.load_state_dict(checkpoint['tts_engine'])
        # self.vocoder.load_state_dict(checkpoint['vocoder'])
        
        # For now, we'll just initialize with random weights
        print("Using randomly initialized weights (no pretrained weights found)")
    
    def detect_emotion(self, audio_path):
        """Detect emotion from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            tuple: (emotion_idx, emotion_probs)
        """
        print(f"Detecting emotion from {audio_path}...")
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_from_waveform(waveform, sample_rate)
        
        # Detect emotion
        emotion_idx, emotion_probs = self.ser_model.predict_emotion(features.unsqueeze(0).to(self.device))
        
        # Map emotion index to label
        emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]
        emotion_label = emotion_labels[emotion_idx[0]]
        
        print(f"Detected emotion: {emotion_label} (Confidence: {emotion_probs[0][emotion_idx[0]]*100:.2f}%)")
        return emotion_idx[0], emotion_probs[0]
    
    def text_to_speech(self, text, emotion_idx=None, save_path=None):
        """Convert text to speech with optional emotion control.
        
        Args:
            text: Input text to convert to speech
            emotion_idx: Optional emotion index (0-4)
            save_path: Optional path to save the generated audio
            
        Returns:
            np.ndarray: Generated audio samples
        """
        print(f"Generating speech for: '{text}'")
        
        # Convert emotion index to one-hot vector if provided
        emotion = None
        if emotion_idx is not None:
            emotion = torch.zeros(5, device=self.device)
            emotion[emotion_idx] = 1.0
            
            emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]
            print(f"Using emotion: {emotion_labels[emotion_idx]}")
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_spectrogram = self.tts_engine.generate(text, emotion=emotion)
            
            # Generate waveform
            audio = self.vocoder.generate(mel_spectrogram)
            
        # Convert to numpy and ensure proper shape
        audio = audio.squeeze().cpu().numpy()
        
        # Save to file if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchaudio.save(save_path, torch.FloatTensor(audio).unsqueeze(0), SAMPLE_RATE)
            print(f"Saved generated audio to {save_path}")
        
        return audio
    
    def play_audio(self, audio, sample_rate=SAMPLE_RATE):
        """Play audio using sounddevice.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate in Hz
        """
        print("Playing audio...")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

def main():
    """Main function for the demo."""
    parser = argparse.ArgumentParser(description="Human Voice AI Demo")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the Human Voice AI system.",
                        help="Text to convert to speech")
    parser.add_argument("--emotion", type=int, default=0, choices=range(5),
                        help="Emotion index (0=Neutral, 1=Happy, 2=Sad, 3=Angry, 4=Surprised)")
    parser.add_argument("--detect_emotion", type=str, default=None,
                        help="Path to audio file for emotion detection")
    parser.add_argument("--output", type=str, default="output/generated_speech.wav",
                        help="Path to save generated speech")
    parser.add_argument("--play", action="store_true",
                        help="Play the generated audio")
    
    args = parser.parse_args()
    
    # Initialize the demo
    print("Initializing Human Voice AI...")
    hva = HumanVoiceAI()
    
    # Detect emotion if audio path is provided
    if args.detect_emotion:
        emotion_idx, _ = hva.detect_emotion(args.detect_emotion)
        args.emotion = emotion_idx
    
    # Generate speech
    audio = hva.text_to_speech(args.text, args.emotion, args.output)
    
    # Play audio if requested
    if args.play:
        hva.play_audio(audio)
    
    print("Done!")

if __name__ == "__main__":
    main()
