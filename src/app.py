#!/usr/bin/env python3
"""
Streamlit web interface for the Emotion-Aware Voice Agent.
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import custom modules
from human_voice_ai.audio.streaming import AudioStream, AudioProcessor
from human_voice_ai.policy.rl_policy import RLPolicy
from human_voice_ai.interpretability.shap_explainer import SERExplainer

# Set page config
st.set_page_config(
    page_title="Emotion-Aware Voice Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
</style>
""", unsafe_allow_html=True)

# Default emotion classes (can be overridden by config)
DEFAULT_EMOTIONS = [
    "neutral", "happy", "sad", "angry", 
    "fear", "disgust", "surprise", "calm"
]

class VoiceAgentApp:
    """Main application class for the Emotion-Aware Voice Agent."""
    
    def __init__(self):
        """Initialize the application."""
        self.audio_stream = None
        self.audio_processor = None
        self.rl_policy = None
        self.shap_explainer = None
        self.emotion_history = []
        self.audio_buffer = np.array([])
        self.class_names = DEFAULT_EMOTIONS
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        if 'emotion_history' not in st.session_state:
            st.session_state.emotion_history = []
        if 'action_history' not in st.session_state:
            st.session_state.action_history = []
    
    def setup_models(self):
        """Initialize the machine learning models."""
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Initialize RL policy (dummy for now)
        state_dim = len(self.class_names)
        action_dim = 5  # Number of possible actions
        self.rl_policy = RLPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            device=device
        )
        
        # Load pre-trained models here
        # self.audio_processor = AudioProcessor(ser_model, device=device)
        # self.shap_explainer = SERExplainer(ser_model, background_data, self.class_names, device)
    
    def start_recording(self):
        """Start the audio stream."""
        if self.audio_stream is None:
            self.audio_stream = AudioStream(callback=self.audio_callback)
            self.audio_stream.start()
            st.session_state.recording = True
            st.experimental_rerun()
    
    def stop_recording(self):
        """Stop the audio stream."""
        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream = None
            st.session_state.recording = False
            st.experimental_rerun()
    
    def audio_callback(self, audio_chunk: np.ndarray):
        """Process an audio chunk from the stream.
        
        Args:
            audio_chunk: Audio data chunk (n_samples, n_channels)
        """
        # For now, just simulate emotion detection
        # In a real app, this would use the audio_processor
        if np.random.random() > 0.5:  # Simulate some processing delay
            emotion = np.random.randint(0, len(self.class_names))
            st.session_state.emotion_history.append(emotion)
            if len(st.session_state.emotion_history) > 100:
                st.session_state.emotion_history = st.session_state.emotion_history[-100:]
    
    def render_status_bar(self):
        """Render the status bar."""
        status_text = "🔴 RECORDING" if st.session_state.recording else "🟢 IDLE"
        status_class = "recording" if st.session_state.recording else "idle"
        st.markdown(f"""
            <div class="status {status_class}">
                <h3>{status_text}</h3>
            </div>
        """, unsafe_allow_html=True)
    
    def render_emotion_distribution(self):
        """Render the emotion distribution chart."""
        if not st.session_state.emotion_history:
            st.info("No emotion data available. Start recording to see analysis.")
            return
        
        # Calculate emotion distribution
        hist = np.bincount(st.session_state.emotion_history, minlength=len(self.class_names))
        total = max(1, hist.sum())
        
        # Create two columns for the chart and metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(self.class_names))
            bars = ax.bar(x, hist / total * 100, color='#2196F3')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 5:  # Only show label if there's enough space
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%',
                            ha='center', va='bottom')
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Emotion Distribution')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            # Show dominant emotion
            dominant_idx = hist.argmax()
            dominant_emoji = self.get_emotion_emoji(dominant_idx)
            st.markdown(f"""
                <div class="metric-box">
                    <h3>Dominant Emotion</h3>
                    <h1>{dominant_emoji} {self.class_names[dominant_idx].title()}</h1>
                    <p>{hist[dominant_idx]} samples ({hist[dominant_idx]/total*100:.1f}%)</p>
                </div>
            """, unsafe_append_html=True)
    
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
            7: "😌"   # calm
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
                st.experimental_rerun()
    
    def render_audio_waveform(self):
        """Render the live audio waveform."""
        if not st.session_state.recording and not st.session_state.emotion_history:
            return
        
        # Create a simple waveform visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        
        if st.session_state.recording:
            # Simulate live waveform
            t = np.linspace(0, 2*np.pi*5, 100)
            y = np.sin(t + time.time() * 5) * 0.5 + 0.5
            ax.plot(y, color='#2196F3', linewidth=1.5)
        
        ax.axis('off')
        ax.margins(x=0)
        
        # Convert to image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
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
                minlength=len(self.class_names)
            )
            action = self.rl_policy.select_action(state)
            
            # Map action to response
            action_map = [
                ("🤖 Generate Response", f"Generating a response for {emotion_name} emotion..."),
                ("📝 Take Notes", f"Taking notes about {emotion_name} emotion..."),
                ("🔔 Set Reminder", f"Setting a reminder to follow up on {emotion_name} emotion..."),
                ("📊 Show Analytics", f"Showing analytics for {emotion_name} emotion..."),
                ("❓ Get Help", f"Getting help for {emotion_name} emotion...")
            ]
            
            # Display action buttons
            for i, (label, _) in enumerate(action_map):
                if st.button(f"{label}", key=f"action_{i}"):
                    st.session_state.action_history.append({
                        'action': i,
                        'emotion': current_emotion,
                        'timestamp': time.time(),
                        'message': action_map[i][1]
                    })
                    st.experimental_rerun()
    
    def render_action_history(self):
        """Render the history of actions taken."""
        if not hasattr(st.session_state, 'action_history') or not st.session_state.action_history:
            return
        
        st.subheader("Action History")
        
        for i, action in enumerate(reversed(st.session_state.action_history[-5:])):  # Show last 5 actions
            emoji = self.get_emotion_emoji(action['emotion'])
            timestamp = time.strftime("%H:%M:%S", time.localtime(action['timestamp']))
            
            with st.expander(f"{timestamp} - {emoji} {action['message']}"):
                st.write(f"Action: {action['action']}")
                st.write(f"Emotion: {self.class_names[action['emotion']].title()}")
    
    def run(self):
        """Run the Streamlit application."""
        st.title("🎙️ Emotion-Aware Voice Agent")
        
        # Initialize models
        if 'models_initialized' not in st.session_state:
            with st.spinner("Initializing models..."):
                self.setup_models()
                st.session_state.models_initialized = True
        
        # Main layout
        self.render_status_bar()
        self.render_controls()
        
        # Two main columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Live Audio")
            self.render_audio_waveform()
            
            st.subheader("Emotion Analysis")
            self.render_emotion_distribution()
        
        with col2:
            self.render_action_buttons()
            self.render_action_history()


if __name__ == "__main__":
    # Initialize the app
    if 'app' not in st.session_state:
        st.session_state.app = VoiceAgentApp()
    
    # Run the app
    st.session_state.app.run()
