#!/usr/bin/env python3
"""
Simplified Streamlit web interface for real-time audio emotion detection.
"""

import os
import sys
import time
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import queue
import sounddevice as sd

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Import custom modules
from human_voice_ai.policy.rl_policy import RLPolicy

# Set page config
st.set_page_config(
    page_title="Voice AI - Real Audio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .audio-level {
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        height: 30px;
        border-radius: 15px;
        margin: 10px 0;
        overflow: hidden;
    }
    .level-fill {
        height: 100%;
        background: white;
        border-radius: 15px;
        transition: width 0.2s ease;
    }
    .status-recording {
        background-color: #ffebee;
        color: #c62828;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .status-idle {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Emotion classes
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "calm"]
EMOTION_EMOJIS = ["😐", "😊", "😢", "😠", "😨", "😲", "😌"]

class SimpleAudioProcessor:
    """Simplified real-time audio processor."""
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.is_recording = False
        self.audio_level = 0.0
        self.audio_data = []
        self.stream = None
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback function."""
        if self.is_recording:
            audio_chunk = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()
            
            # Calculate RMS level
            self.audio_level = float(np.sqrt(np.mean(audio_chunk**2)) * 1000)
            
            # Store audio data
            self.audio_data.extend(audio_chunk)
            
            # Keep only last 5 seconds
            max_samples = self.sample_rate * 5
            if len(self.audio_data) > max_samples:
                self.audio_data = self.audio_data[-max_samples:]
    
    def start_recording(self):
        """Start audio recording."""
        try:
            self.is_recording = True
            self.audio_data = []
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size
            )
            self.stream.start()
            return True
        except Exception as e:
            st.error(f"Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording."""
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
            self.stream = None
    
    def get_audio_level(self):
        """Get current audio level (0-100)."""
        return min(self.audio_level, 100)
    
    def analyze_emotion(self):
        """Simple emotion analysis."""
        if len(self.audio_data) < self.sample_rate:  # Need at least 1 second
            return 0  # neutral
        
        # Get last 2 seconds of audio
        recent_audio = np.array(self.audio_data[-2*self.sample_rate:])
        
        # Calculate features
        rms = np.sqrt(np.mean(recent_audio**2))
        zero_crossings = np.sum(np.diff(np.signbit(recent_audio)))
        
        # Simple classification
        if rms > 0.05 and zero_crossings > 800:
            return 3  # angry
        elif rms > 0.02 and zero_crossings > 600:
            return 1  # happy
        elif rms < 0.01:
            return 2  # sad
        elif zero_crossings > 1000:
            return 4  # fear
        else:
            return 0  # neutral

class VoiceApp:
    """Main Voice AI application."""
    
    def __init__(self):
        self.audio_processor = SimpleAudioProcessor()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state."""
        if "recording" not in st.session_state:
            st.session_state.recording = False
        if "emotion_history" not in st.session_state:
            st.session_state.emotion_history = []
        if "audio_level" not in st.session_state:
            st.session_state.audio_level = 0.0
        if "last_analysis" not in st.session_state:
            st.session_state.last_analysis = time.time()
    
    def start_recording(self):
        """Start recording."""
        if self.audio_processor.start_recording():
            st.session_state.recording = True
            st.success("🎤 Recording started!")
        else:
            st.error("❌ Failed to start recording")
    
    def stop_recording(self):
        """Stop recording."""
        self.audio_processor.stop_recording()
        st.session_state.recording = False
        st.success("⏹️ Recording stopped")
    
    def update_analysis(self):
        """Update emotion analysis."""
        if st.session_state.recording:
            # Update audio level
            st.session_state.audio_level = self.audio_processor.get_audio_level()
            
            # Analyze emotion every 3 seconds
            if time.time() - st.session_state.last_analysis > 3.0:
                emotion = self.audio_processor.analyze_emotion()
                st.session_state.emotion_history.append(emotion)
                st.session_state.last_analysis = time.time()
                
                # Keep history reasonable
                if len(st.session_state.emotion_history) > 50:
                    st.session_state.emotion_history = st.session_state.emotion_history[-50:]
    
    def render_status(self):
        """Render status section."""
        if st.session_state.recording:
            level = st.session_state.audio_level
            st.markdown(f"""
                <div class="status-recording">
                    🔴 RECORDING - Audio Level: {level:.1f}%
                </div>
            """, unsafe_allow_html=True)
            
            # Audio level bar
            st.markdown(f"""
                <div class="audio-level">
                    <div class="level-fill" style="width: {level}%;"></div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="status-idle">
                    🟢 IDLE - Click Start Recording to begin
                </div>
            """, unsafe_allow_html=True)
    
    def render_controls(self):
        """Render control buttons."""
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.recording:
                if st.button("🎤 Start Recording", type="primary", use_container_width=True):
                    self.start_recording()
            else:
                if st.button("⏹️ Stop Recording", type="primary", use_container_width=True):
                    self.stop_recording()
        
        with col2:
            if st.button("🧹 Clear Data", use_container_width=True):
                st.session_state.emotion_history = []
                st.success("Data cleared!")
    
    def render_emotions(self):
        """Render emotion analysis."""
        if not st.session_state.emotion_history:
            st.info("Start recording to see emotion analysis!")
            return
        
        # Calculate distribution
        history = st.session_state.emotion_history
        emotion_counts = [history.count(i) for i in range(len(EMOTIONS))]
        total = max(1, sum(emotion_counts))
        
        # Show current emotion
        if history:
            current_emotion = history[-1]
            emoji = EMOTION_EMOJIS[current_emotion]
            emotion_name = EMOTIONS[current_emotion]
            
            st.markdown(f"""
                ### Current Emotion: {emoji} {emotion_name.title()}
                *Based on last 3 seconds of audio*
            """)
        
        # Show distribution chart
        if len(history) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            percentages = [count/total*100 for count in emotion_counts]
            bars = ax.bar(EMOTIONS, percentages, color='#2196F3')
            
            # Add percentage labels
            for bar, pct in zip(bars, percentages):
                if pct > 5:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{pct:.1f}%', ha='center', va='bottom')
            
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Emotion Distribution')
            ax.set_ylim(0, max(percentages) * 1.2 if percentages else 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
        
        # Show recent history
        if len(history) >= 3:
            st.markdown("### Recent Emotions:")
            recent = history[-5:]  # Last 5 emotions
            emotion_text = " → ".join([f"{EMOTION_EMOJIS[e]} {EMOTIONS[e]}" for e in recent])
            st.markdown(emotion_text)
    
    def run(self):
        """Run the app."""
        st.title("🎤 Voice AI - Real Audio Emotion Detection")
        st.markdown("*This version processes actual microphone input for real-time emotion analysis.*")
        
        # Update analysis
        self.update_analysis()
        
        # Status
        self.render_status()
        
        # Controls
        st.subheader("Controls")
        self.render_controls()
        
        # Emotions
        st.subheader("Emotion Analysis")
        self.render_emotions()
        
        # Auto-refresh when recording
        if st.session_state.recording:
            time.sleep(0.5)
            st.rerun()

if __name__ == "__main__":
    app = VoiceApp()
    app.run() 