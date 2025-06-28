#!/usr/bin/env python3
"""
Test script for the RL pipeline.

This script verifies that all components of the RL pipeline are working correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.human_voice_ai.policy.rl_policy import EmotionPolicy
from src.human_voice_ai.rl.agent import PolicyNetwork
from src.human_voice_ai.rl.environment import VoiceAIEnvironment
from src.human_voice_ai.tts.tts_engine import TtsEngine

def test_rl_pipeline():
    print("Testing RL pipeline...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test Policy Network
    print("\nTesting Policy Network...")
    state_dim = 256  # Example state dimension
    action_dim = 5    # Example number of emotions
    policy_net = PolicyNetwork(state_dim, action_dim).to(device)
    
    # Test forward pass
    test_state = torch.randn(1, state_dim).to(device)
    action_probs, state_value = policy_net(test_state)
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"State value shape: {state_value.shape}")
    
    # Test EmotionPolicy
    print("\nTesting EmotionPolicy...")
    config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon': 0.9,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 10000
    }
    
    policy = EmotionPolicy(config)
    print("EmotionPolicy initialized successfully")
    
    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = policy.select_action(state)
    print(f"Selected action: {action}")
    
    # Test environment (without audio)
    print("\nTesting Environment (without audio)...")
    env_config = {
        'use_audio': False,  # Disable audio for testing
        'max_turns': 5,
        'state_dim': state_dim
    }
    
    env = VoiceAIEnvironment(env_config)
    print("Environment initialized successfully")
    
    # Run a simple episode
    print("\nRunning test episode...")
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        print(f"Step - Action: {action}, Reward: {reward:.2f}, Done: {done}")
    
    print(f"Test episode completed. Total reward: {total_reward:.2f}")
    print("\nRL pipeline test completed successfully!")

if __name__ == "__main__":
    test_rl_pipeline()
