#!/usr/bin/env python3
"""
Stable Voice AI with file-based audio processing and pre-trained models.
Optimized for Mac M2 with 8GB RAM.
"""

import os
import sys
import time
import tempfile
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import librosa
import sounddevice as sd
import soundfile as sf
from transformers import pipeline
import gc

# Add project root to path
sys.path.append(str(Path(__file__).parent.absolute()))

# Set page config
st.set_page_config(
    page_title="Voice AI - Emotion Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Google DeepMind-inspired CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
    }
    
    .hero-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .emotion-card {
        background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(26, 115, 232, 0.3);
        margin: 1rem 0;
        text-align: center;
        transform: translateY(0);
        transition: transform 0.3s ease;
    }
    
    .emotion-card:hover {
        transform: translateY(-4px);
    }
    
    .analytics-panel {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #1a73e8 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: #64748b;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class EmotionDetector:
    """Advanced emotion detection using pre-trained models."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.temp_dir = tempfile.mkdtemp()
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize pre-trained emotion detection model."""
        try:
            # Try to load emotion classification model
            st.info("🔄 Loading emotion detection model...")
            self.emotion_pipeline = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=-1  # Use CPU for stability
            )
            st.success("✅ Emotion detection model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Using fallback emotion detection: {e}")
            self.emotion_pipeline = None
    
    def record_audio(self, duration=3.0):
        """Record audio to file (stable approach)."""
        try:
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Save to temporary file
            temp_file = os.path.join(self.temp_dir, f"recording_{int(time.time())}.wav")
            sf.write(temp_file, audio_data, self.sample_rate)
            
            return temp_file, audio_data
        except Exception as e:
            st.error(f"Recording failed: {e}")
            return None, None
    
    def extract_features(self, audio_data):
        """Extract comprehensive audio features."""
        features = {}
        
        try:
            # Basic features
            features['rms'] = float(np.sqrt(np.mean(audio_data**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_data)))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}'] = float(np.mean(mfccs[i]))
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            
        except Exception as e:
            st.warning(f"Feature extraction warning: {e}")
        
        return features
    
    def classify_emotion(self, audio_file, audio_data):
        """Classify emotion using pre-trained model."""
        try:
            if self.emotion_pipeline and audio_file:
                # Use HuggingFace model
                result = self.emotion_pipeline(audio_file)
                
                # Map to 1-5 scale
                emotion_map = {
                    'sad': 1, 'angry': 1, 'fear': 1, 'disgust': 1,  # Very Negative
                    'calm': 2,  # Slightly Negative
                    'neutral': 3,  # Neutral
                    'happy': 4,  # Slightly Positive
                    'excited': 5, 'joy': 5, 'surprise': 5  # Very Positive
                }
                
                if result:
                    emotion_label = result[0]['label'].lower()
                    confidence = result[0]['score']
                    emotion_score = emotion_map.get(emotion_label, 3)
                    
                    return {
                        'score': emotion_score,
                        'confidence': confidence,
                        'label': emotion_label,
                        'raw_result': result
                    }
            
            # Fallback to feature-based classification
            features = self.extract_features(audio_data)
            return self.fallback_classification(features)
            
        except Exception as e:
            st.warning(f"Classification warning: {e}")
            features = self.extract_features(audio_data)
            return self.fallback_classification(features)
    
    def fallback_classification(self, features):
        """Fallback emotion classification based on audio features."""
        rms = features.get('rms', 0)
        zcr = features.get('zero_crossing_rate', 0)
        spectral_centroid = features.get('spectral_centroid', 0)
        
        # Enhanced rule-based classification
        if rms > 0.15 and zcr > 0.1:
            score = 1  # Very Negative (Angry/Upset)
            label = "angry"
            confidence = min(rms * 5, 1.0)
        elif rms > 0.08 and spectral_centroid > 2000:
            score = 5  # Very Positive (Happy/Excited)
            label = "happy"
            confidence = min(rms * 4, 1.0)
        elif rms < 0.03:
            score = 2  # Slightly Negative (Sad/Low energy)
            label = "sad"
            confidence = 1.0 - rms * 10
        elif zcr > 0.08:
            score = 4  # Slightly Positive (Engaged/Interested)
            label = "engaged"
            confidence = min(zcr * 8, 1.0)
        else:
            score = 3  # Neutral
            label = "neutral"
            confidence = 0.7
        
        return {
            'score': score,
            'confidence': confidence,
            'label': label,
            'raw_result': None
        }

class VoiceAIApp:
    """Main Voice AI Application with modern UI."""
    
    def __init__(self):
        self.detector = EmotionDetector()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state."""
        if "emotion_history" not in st.session_state:
            st.session_state.emotion_history = []
        if "current_emotion" not in st.session_state:
            st.session_state.current_emotion = None
        if "recording_count" not in st.session_state:
            st.session_state.recording_count = 0
    
    def render_hero(self):
        """Render hero section."""
        st.markdown("""
            <div class="hero-container">
                <h1 class="title-gradient">Voice AI Emotion Detection</h1>
                <p class="subtitle">Advanced emotion analysis using state-of-the-art AI models</p>
            </div>
        """, unsafe_allow_html=True)
    
    def render_controls(self):
        """Render audio controls."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div class="analytics-panel">
                    <h3 style="text-align: center; margin-bottom: 1rem;">🎤 Voice Recording</h3>
                    <p style="text-align: center; color: #64748b;">Click the button below to record 5 seconds of audio</p>
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎤 Record 5 Seconds", key="record_btn", type="primary", use_container_width=True):
                self.record_and_analyze()
            
            if st.button("🧹 Clear History", key="clear_btn", use_container_width=True):
                st.session_state.emotion_history = []
                st.session_state.current_emotion = None
                st.rerun()
    
    def record_and_analyze(self):
        """Record audio and perform analysis (5-second recording)."""
        with st.spinner("🎵 Recording 5 seconds of audio..."):
            audio_file, audio_data = self.detector.record_audio(duration=5.0)
        
        if audio_file and audio_data is not None:
            with st.spinner("🧠 Analyzing emotion..."):
                # Classify emotion
                emotion_result = self.detector.classify_emotion(audio_file, audio_data.flatten())
                
                # Store results
                st.session_state.current_emotion = emotion_result
                st.session_state.emotion_history.append(emotion_result)
                st.session_state.recording_count += 1
                
                # Clean up temp file
                try:
                    os.remove(audio_file)
                except:
                    pass
                
                # Force garbage collection
                gc.collect()
            
            st.success("✅ Analysis complete!")
            st.rerun()
        else:
            st.error("❌ Recording failed. Please try again.")
    
    def render_current_emotion(self):
        """Render current emotion analysis."""
        if not st.session_state.current_emotion:
            st.markdown("""
                <div class="analytics-panel">
                    <h3 style="text-align: center;">📊 Emotion Analysis</h3>
                    <p style="text-align: center; color: #64748b;">Record audio to see detailed emotion analysis</p>
                </div>
            """, unsafe_allow_html=True)
            return
        
        emotion = st.session_state.current_emotion
        score = emotion['score']
        confidence = emotion['confidence']
        label = emotion['label']
        
        # Emotion mapping
        score_labels = {
            1: "Very Negative", 2: "Slightly Negative", 3: "Neutral",
            4: "Slightly Positive", 5: "Very Positive"
        }
        
        score_emojis = {
            1: "😠", 2: "😔", 3: "😐", 4: "🙂", 5: "😊"
        }
        
        score_colors = {
            1: "#ef4444", 2: "#f97316", 3: "#64748b", 4: "#10b981", 5: "#22c55e"
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
                <div class="emotion-card">
                    <h2>{score_emojis[score]} {score_labels[score]}</h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0;">{score}/5</h1>
                    <p>Detected: <strong>{label.title()}</strong></p>
                    <p>Confidence: <strong>{confidence:.1%}</strong></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': score_colors[score]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    def render_history(self):
        """Render emotion history."""
        if len(st.session_state.emotion_history) < 2:
            return
        
        st.markdown("""
            <div class="analytics-panel">
                <h3>📈 Emotion Timeline</h3>
                <p style="color: #64748b;">Your emotion progression over time:</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create timeline chart
        history = st.session_state.emotion_history[-10:]  # Last 10 recordings
        scores = [e['score'] for e in history]
        confidences = [e['confidence'] for e in history]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            mode='lines+markers',
            name='Emotion Score',
            line=dict(color='#1a73e8', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Emotion Score Timeline",
            xaxis_title="Recording #",
            yaxis_title="Score (1-5)",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0.5, 5.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the application."""
        self.render_hero()
        self.render_controls()
        self.render_current_emotion()
        self.render_history()
        
        # Show recording count
        if st.session_state.recording_count > 0:
            st.sidebar.success(f"🎤 Recordings: {st.session_state.recording_count}")
            st.sidebar.info("💡 Tip: Try different emotions to see how the AI responds!")

if __name__ == "__main__":
    app = VoiceAIApp()
    app.run() 