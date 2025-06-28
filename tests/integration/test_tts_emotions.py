"""
Integration test for TTS engine with emotion control.

This script tests the TTS engine's ability to generate speech with different emotions
and verifies that the output files are created correctly.
"""

import os
import tempfile
import torch
import torchaudio
from pathlib import Path
from human_voice_ai.tts.tts_engine import TtsEngine, TTSConfig


def test_tts_emotion_generation():
    """Test TTS generation with different emotions."""
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize TTS engine with test config
        config = TTSConfig(
            model_name="tts_models/en/ljspeech/glow-tts",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tts = TtsEngine(config=config)

        # Test text
        test_text = "Hello, this is a test of the emotion-controlled TTS system."

        # Get available emotions
        emotions = tts.get_available_emotions()
        print("\nTesting TTS with the following emotions:")

        # Test each emotion
        for emotion_name, emotion_id in emotions.items():
            print(
                f"\nGenerating speech with emotion: {emotion_name} (ID: {emotion_id})"
            )

            # Generate output path
            output_path = Path(temp_dir) / f"test_emotion_{emotion_name}.wav"

            try:
                # Generate speech
                audio, sample_rate = tts.synthesize(
                    text=test_text, emotion_id=emotion_id
                )

                # Save the audio
                torchaudio.save(output_path, audio, sample_rate)

                # Verify file was created and has content
                assert (
                    output_path.exists()
                ), f"Output file not created for {emotion_name}"
                assert (
                    output_path.stat().st_size > 0
                ), f"Empty output file for {emotion_name}"

                print(
                    f"  ✓ Successfully generated {output_path.name} ({output_path.stat().st_size} bytes)"
                )

            except Exception as e:
                print(f"  ✗ Error generating {emotion_name}: {str(e)}")
                raise

    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    test_tts_emotion_generation()
