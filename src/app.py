#!/usr/bin/env python3
"""
Streamlit web interface for the Emotion-Aware Voice Agent with real audio processing.
"""

import os
import sys
import time
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import io
from PIL import Image
import threading
import queue
import sounddevice as sd
import librosa

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import custom modules
from human_voice_ai.policy.rl_policy import RLPolicy

# Set page config
st.set_page_config(
    page_title="Emotion-Aware Voice Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main { 
        padding: 2rem; 
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
    }
    .metric-box { 
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        background-color: #f8f9fa;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .emotion-bar {
        height: 24px;
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        border-radius: 12px;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .emotion-fill {
        height: 100%;
        background: linear-gradient(90deg, #2196F3, #00BCD4);
        transition: width 0.3s ease;
    }
    .emotion-label {
        position: absolute;
        width: 100%;
        text-align: center;
        color: white;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    .status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .recording {
        background-color: #ffebee;
        color: #c62828;
    }
    .idle {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    .audio-level {
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        height: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Default emotion classes (can be overridden by config)
DEFAULT_EMOTIONS = [
    "neutral",
    "happy", 
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
    "calm",
]


class RealTimeAudioProcessor:
    """Real-time audio processor for emotion detection."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.current_level = 0.0
        self.audio_buffer = []
        self.stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        if self.is_recording:
            # Calculate audio level (RMS)
            audio_data = indata[:, 0] if len(indata.shape) > 1 else indata
            self.current_level = np.sqrt(np.mean(audio_data**2))
            
            # Store audio for processing
            self.audio_buffer.extend(audio_data)
            
            # Keep buffer to reasonable size (5 seconds)
            max_buffer_size = self.sample_rate * 5
            if len(self.audio_buffer) > max_buffer_size:
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
            
            # Put in queue for processing
            try:
                self.audio_queue.put_nowait(audio_data.copy())
            except queue.Full:
                pass  # Skip if queue is full
    
    def start_recording(self):
        """Start audio recording."""
        try:
            self.is_recording = True
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=None  # Use default input device
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording."""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_audio_level(self):
        """Get current audio level."""
        return min(self.current_level * 100, 100)  # Scale to 0-100
    
    def get_recent_audio(self, duration_seconds=2):
        """Get recent audio data for processing."""
        if len(self.audio_buffer) < self.sample_rate * duration_seconds:
            return None
        
        # Get last N seconds of audio
        samples_needed = int(self.sample_rate * duration_seconds)
        return np.array(self.audio_buffer[-samples_needed:])
    
    def analyze_emotion(self, audio_data):
        """Simple emotion analysis based on audio features."""
        if audio_data is None or len(audio_data) == 0:
            return 0  # neutral
        
        try:
            # Extract basic audio features
            rms = np.sqrt(np.mean(audio_data**2))
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate))
            
            # Simple rule-based emotion classification
            # This is a placeholder - in a real system you'd use a trained model
            if rms > 0.1 and zero_crossings > 1000:
                return 3  # angry
            elif rms > 0.05 and spectral_centroid > 2000:
                return 1  # happy
            elif rms < 0.02:
                return 2  # sad
            elif zero_crossings > 1500:
                return 4  # fear
            else:
                return 0  # neutral
                
        except Exception as e:
            print(f"Error in emotion analysis: {e}")
            return 0  # neutral


class VoiceAgentApp:
    """Main application class for the Emotion-Aware Voice Agent."""

    def __init__(self):
        """Initialize the application."""
        self.audio_processor = RealTimeAudioProcessor()
        self.rl_policy = None
        self.class_names = DEFAULT_EMOTIONS
        self.setup_session_state()
        self.setup_models()

    def setup_session_state(self):
        """Initialize Streamlit session state variables."""
        if "recording" not in st.session_state:
            st.session_state.recording = False
        if "emotion_history" not in st.session_state:
            st.session_state.emotion_history = []
        if "action_history" not in st.session_state:
            st.session_state.action_history = []
        if "audio_level" not in st.session_state:
            st.session_state.audio_level = 0.0
        if "last_emotion_time" not in st.session_state:
            st.session_state.last_emotion_time = 0

    def setup_models(self):
        """Initialize the machine learning models."""
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        # Initialize RL policy
        state_dim = len(self.class_names)
        action_dim = 5  # Number of possible actions
        self.rl_policy = RLPolicy(
            state_dim=state_dim, action_dim=action_dim, hidden_dim=128, device=device
        )

    def start_recording(self):
        """Start the audio stream."""
        success = self.audio_processor.start_recording()
        if success:
            st.session_state.recording = True
            st.success("🎤 Recording started! Please speak...")
        else:
            st.error("❌ Failed to start recording. Please check microphone permissions.")
        st.rerun()

    def stop_recording(self):
        """Stop the audio stream."""
        self.audio_processor.stop_recording()
        st.session_state.recording = False
        st.success("⏹️ Recording stopped.")
        st.rerun()

    def update_emotion_analysis(self):
        """Update emotion analysis from recent audio."""
        if st.session_state.recording:
            # Update audio level
            st.session_state.audio_level = self.audio_processor.get_audio_level()
            
            # Analyze emotion every 2 seconds
            current_time = time.time()
            if current_time - st.session_state.last_emotion_time > 2.0:
                recent_audio = self.audio_processor.get_recent_audio(duration_seconds=2)
                if recent_audio is not None:
                    emotion = self.audio_processor.analyze_emotion(recent_audio)
                    st.session_state.emotion_history.append(emotion)
                    st.session_state.last_emotion_time = current_time
                    
                    # Keep history manageable
                    if len(st.session_state.emotion_history) > 100:
                        st.session_state.emotion_history = st.session_state.emotion_history[-100:]

    def render_status_bar(self):
        """Render the status bar."""
        status_text = "🔴 RECORDING" if st.session_state.recording else "🟢 IDLE"
        status_class = "recording" if st.session_state.recording else "idle"
        
        # Add audio level indicator
        level_indicator = ""
        if st.session_state.recording:
            level = st.session_state.audio_level
            level_indicator = f" (Level: {level:.1f}%)"
        
        st.markdown(
            f"""
            <div class="status {status_class}">
                <h3>{status_text}{level_indicator}</h3>
            </div>
        """,
            unsafe_allow_html=True,
        )
        
        # Show audio level bar
        if st.session_state.recording:
            level = st.session_state.audio_level
            st.markdown(
                f"""
                <div class="audio-level">
                    <div style="width: {level}%; height: 100%; background: white; border-radius: 10px; transition: width 0.3s;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    def render_emotion_distribution(self):
        """Render the emotion distribution chart."""
        if not st.session_state.emotion_history:
            st.info("No emotion data available. Start recording to see analysis.")
            return

        # Calculate emotion distribution
        hist = np.bincount(
            st.session_state.emotion_history, minlength=len(self.class_names)
        )
        total = max(1, hist.sum())

        # Create two columns for the chart and metrics
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(self.class_names))
            bars = ax.bar(x, hist / total * 100, color="#2196F3")

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Only show label if there's enough space
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                    )

            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha="right")
            ax.set_ylabel("Percentage (%)")
            ax.set_title("Emotion Distribution")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            # Show dominant emotion
            dominant_idx = hist.argmax()
            dominant_emoji = self.get_emotion_emoji(dominant_idx)
            st.markdown(
                f"""
                <div class="metric-box">
                    <h3>Dominant Emotion</h3>
                    <h1>{dominant_emoji} {self.class_names[dominant_idx].title()}</h1>
                    <p>{hist[dominant_idx]} samples ({hist[dominant_idx]/total*100:.1f}%)</p>
                </div>
            """,
                unsafe_allow_html=True,
            )

    def get_emotion_emoji(self, emotion_idx: int) -> str:
        """Get an emoji for the given emotion index."""
        emoji_map = {
            0: "😐",  # neutral
            1: "😊",  # happy
            2: "😢",  # sad
            3: "😠",  # angry
            4: "😨",  # fear
            5: "🤢",  # disgust
            6: "😲",  # surprise
            7: "😌",  # calm
        }
        return emoji_map.get(emotion_idx % len(emoji_map), "❓")

    def render_controls(self):
        """Render the control buttons."""
        col1, col2 = st.columns(2)

        with col1:
            if not st.session_state.recording:
                if st.button("🎤 Start Recording", key="start_btn", type="primary"):
                    self.start_recording()
            else:
                if st.button("⏹️ Stop Recording", key="stop_btn", type="primary"):
                    self.stop_recording()

        with col2:
            if st.button("🧹 Clear Data", key="clear_btn"):
                st.session_state.emotion_history = []
                st.session_state.action_history = []
                st.rerun()

    def render_audio_waveform(self):
        """Render the live audio waveform."""
        if not st.session_state.recording and not st.session_state.emotion_history:
            return

        # Create a simple waveform visualization
        fig, ax = plt.subplots(figsize=(10, 2))

        if st.session_state.recording:
            # Show real-time audio level as waveform
            level = st.session_state.audio_level / 100.0
            t = np.linspace(0, 2 * np.pi * 3, 100)
            y = np.sin(t + time.time() * 10) * level + 0.5
            ax.plot(y, color="#2196F3", linewidth=2)
            ax.fill_between(range(len(y)), 0, y, alpha=0.3, color="#2196F3")
        else:
            # Show static waveform when not recording
            y = np.zeros(100) + 0.5
            ax.plot(y, color="#cccccc", linewidth=1)

        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.margins(x=0)

        # Convert to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        img_buf.seek(0)
        img = Image.open(img_buf)

        # Display with caption
        st.image(img, use_column_width=True)
        plt.close(fig)

    def render_action_buttons(self):
        """Render the action buttons based on the current state."""
        st.subheader("Actions")

        if not st.session_state.emotion_history:
            st.info("Record some audio to enable actions")
            return
        
        # Get current emotion state
        if st.session_state.emotion_history:
            current_emotion = st.session_state.emotion_history[-1]
            emotion_name = self.class_names[current_emotion].title()

            # Get recommended action from RL policy
            state = np.bincount(
                st.session_state.emotion_history[-10:],  # Last 10 emotions
                minlength=len(self.class_names),
            )
            action = self.rl_policy.select_action(state)

            # Map action to response
            action_map = [
                ("🤖 Generate Response", f"Generating a response for {emotion_name} emotion..."),
                ("📝 Take Notes", f"Taking notes about {emotion_name} emotion..."),
                ("🔔 Set Reminder", f"Setting a reminder to follow up on {emotion_name} emotion..."),
                ("📊 Show Analytics", f"Showing analytics for {emotion_name} emotion..."),
                ("❓ Get Help", f"Getting help for {emotion_name} emotion..."),
            ]

            # Display action buttons
            for i, (label, message) in enumerate(action_map):
                if st.button(f"{label}", key=f"action_{i}"):
                    st.session_state.action_history.append({
                        "action": i,
                        "emotion": current_emotion,
                        "timestamp": time.time(),
                        "message": message,
                    })
                    st.success(message)
                    st.rerun()

    def render_action_history(self):
        """Render the action history."""
        if not st.session_state.action_history:
            return

        st.subheader("Action History")
        for i, action in enumerate(reversed(st.session_state.action_history[-5:])):
            emotion_name = self.class_names[action["emotion"]].title()
            emoji = self.get_emotion_emoji(action["emotion"])
            timestamp = time.strftime("%H:%M:%S", time.localtime(action["timestamp"]))
            
            st.markdown(f"**{timestamp}** - {emoji} {emotion_name}: {action['message']}")

    def run(self):
        """Run the Streamlit app."""
        st.title("🎤 Emotion-Aware Voice Agent")
        
        # Update emotion analysis if recording
        self.update_emotion_analysis()

        # Status bar
        self.render_status_bar()

        # Main content
        col1, col2 = st.columns([2, 1])

        with col1:
            # Controls
            self.render_controls()
            
            # Live Audio Waveform
            st.subheader("Live Audio")
            self.render_audio_waveform()

            # Emotion Analysis
            st.subheader("Emotion Analysis")
            self.render_emotion_distribution()

        with col2:
            # Actions
            self.render_action_buttons()
            
            # Action History
            self.render_action_history()

        # Auto-refresh when recording
        if st.session_state.recording:
            time.sleep(0.1)
            st.rerun()


if __name__ == "__main__":
    app = VoiceAgentApp()
    app.run()
