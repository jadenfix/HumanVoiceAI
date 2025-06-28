"""
Test script for TTS emotion generation with visualization and playback.

This script demonstrates the TTS engine's emotion capabilities by:
1. Generating speech samples with different emotions
2. Visualizing the audio waveforms and spectrograms
3. Playing back the generated samples
"""

import os
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
import sounddevice as sd
from tqdm import tqdm

from human_voice_ai.tts.tts_engine import TtsEngine, TTSConfig


def plot_waveform(waveform, sample_rate, title="Waveform", ax=None):
    """Plot waveform of an audio signal."""
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    return ax


def plot_spectrogram(specgram, title=None, ylabel="Frequency (bin)", ax=None):
    """Plot spectrogram of an audio signal."""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 3))

    # Handle different input dimensions
    if specgram.dim() > 3:
        specgram = specgram.squeeze(0)
    if specgram.dim() > 2:
        specgram = specgram[0]  # Take first channel if multi-channel

    # Convert power to dB scale
    specgram_db = torchaudio.transforms.AmplitudeToDB()(specgram)

    # Ensure we have a 2D array for plotting
    if specgram_db.dim() == 1:
        specgram_db = specgram_db.unsqueeze(0)

    # Plot
    img = ax.imshow(
        specgram_db.cpu().numpy(),
        cmap="viridis",
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Frame")

    # Add colorbar
    plt.colorbar(img, ax=ax, format="%+2.0f dB")

    return ax


def play_audio(waveform, sample_rate):
    """Play audio using sounddevice."""
    audio = waveform.numpy()
    if len(audio.shape) > 1 and audio.shape[0] > 1:
        audio = audio[0]  # Just use the first channel

    print(
        f"Playing audio (sample rate: {sample_rate}Hz, duration: {len(audio[0])/sample_rate:.2f}s)"
    )
    sd.play(audio[0], sample_rate)
    sd.wait()


def main():
    # Create output directory
    output_dir = Path("output/tts_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TTS engine
    print("Initializing TTS engine...")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    config = TTSConfig(device=device)
    tts = TtsEngine(config=config)

    # Test text
    test_text = (
        "Hello, this is a test of the emotion-controlled text-to-speech system. "
        "Can you hear the difference in my voice?"
    )

    # Get available emotions
    emotions = tts.get_available_emotions()
    print(f"\nAvailable emotions: {', '.join(emotions.keys())}")

    # Generate samples for each emotion
    print("\nGenerating samples...")
    results = {}

    for emotion_name, emotion_id in tqdm(
        emotions.items(), desc="Generating emotion samples"
    ):
        try:
            # Generate speech
            audio, sample_rate = tts.synthesize(text=test_text, emotion_id=emotion_id)

            # Save audio
            output_path = output_dir / f"tts_emotion_{emotion_name}.wav"
            torchaudio.save(output_path, audio, sample_rate)

            # Generate spectrogram with more robust parameters
            waveform = audio
            n_fft = 2048  # Increased FFT size for better frequency resolution
            win_length = None  # Default to n_fft
            hop_length = 512  # 50% overlap
            n_mels = 80  # Standard number of mel bands

            # Ensure waveform is 2D (channels, samples)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Create mel spectrogram transform
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                center=True,
                pad_mode="reflect",
            ).to(waveform.device)

            # Compute mel spectrogram
            melspec = mel_spectrogram(waveform)

            # Ensure we have a 2D tensor (n_mels, time)
            if melspec.dim() > 2:
                melspec = melspec.squeeze(0)

            results[emotion_name] = {
                "audio": audio,
                "sample_rate": sample_rate,
                "melspec": melspec,
                "path": output_path,
            }

            print(f"  ✓ Generated {output_path.name}")

        except Exception as e:
            print(f"  ✗ Error with {emotion_name}: {str(e)}")

    # Plot comparison of all emotions
    print("\nGenerating comparison plots...")
    fig, axes = plt.subplots(len(emotions), 2, figsize=(15, 3 * len(emotions)))

    for idx, (emotion_name, data) in enumerate(results.items()):
        # Plot waveform
        plot_waveform(
            data["audio"],
            data["sample_rate"],
            title=f"{emotion_name.capitalize()} Waveform",
            ax=axes[idx, 0],
        )

        # Plot spectrogram
        plot_spectrogram(
            data["melspec"],
            title=f"{emotion_name.capitalize()} Mel Spectrogram",
            ax=axes[idx, 1],
        )

    plt.tight_layout()
    plot_path = output_dir / "emotion_comparison.png"
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")

    # Interactive playback
    print("\nInteractive playback:")
    while True:
        print("\nSelect an emotion to play (or 'q' to quit):")
        for i, emotion in enumerate(emotions.keys(), 1):
            print(f"  {i}. {emotion}")
        print("  q. Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(emotions):
                emotion_name = list(emotions.keys())[idx]
                print(f"\nPlaying {emotion_name}...")
                play_audio(
                    results[emotion_name]["audio"], results[emotion_name]["sample_rate"]
                )
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number or 'q' to quit.")

    print("\nTest completed!")
    print(f"All generated files are saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
