"""
Integration tests for the RL pipeline.

These tests verify that the RL components work together correctly.
"""

import os
import sys
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from human_voice_ai.rl.agent import PolicyNetwork, PPOTrainer, ReplayBuffer, Transition
from human_voice_ai.rl.environment import VoiceAIEnvironment, VoiceAIState


@pytest.fixture
def mock_ser_model():
    """Create a mock SER model for testing."""
    model = MagicMock()
    model.return_value = {"logits": torch.randn(1, 5)}  # 5 emotion classes
    return model


@pytest.fixture
def mock_tts_engine():
    """Create a mock TTS engine for testing."""
    engine = MagicMock()
    engine.synthesize.return_value = (np.zeros(16000), 16000)  # 1 second of silence
    return engine


def test_rl_environment_initialization(mock_ser_model, mock_tts_engine):
    """Test that the RL environment initializes correctly."""
    env = VoiceAIEnvironment(
        ser_model=mock_ser_model,
        tts_engine=mock_tts_engine,
        max_turns=10,
        state_dim=256,
        history_length=5
    )
    
    assert env is not None
    assert env.max_turns == 10
    assert env.state_dim == 256
    assert env.history_length == 5


def test_policy_network_forward():
    """Test the forward pass of the policy network."""
    state_dim = 256
    action_dim = 5
    batch_size = 4
    history_length = 5  # This should match the expected history length in the network

    policy_net = PolicyNetwork(state_dim=state_dim, hidden_dim=128, num_actions=action_dim)

    # Create a batch of states with the expected structure
    state = {
        "audio": torch.randn(batch_size, state_dim),
        "text": torch.randn(batch_size, state_dim),
        "emotion_history": torch.zeros(batch_size, 5),  # Current emotion (one-hot encoded)
        "turn_count": torch.randn(batch_size, 1)
    }

    # Test forward pass
    action_logits, state_value = policy_net(state)
    
    assert action_logits.shape == (batch_size, action_dim)
    assert state_value.shape == (batch_size,)


def test_ppo_trainer_initialization():
    """Test that the PPO trainer initializes correctly."""
    state_dim = 256
    action_dim = 5

    policy = PolicyNetwork(state_dim=state_dim, hidden_dim=128, num_actions=action_dim)
    
    trainer = PPOTrainer(
        policy=policy,
        lr=1e-4,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5
    )
    
    assert trainer is not None
    assert trainer.gamma == 0.99
    assert trainer.epsilon == 0.2


def test_replay_buffer():
    """Test the replay buffer functionality."""
    buffer_size = 100
    state_dim = 256
    action_dim = 5
    history_length = 3

    buffer = ReplayBuffer(buffer_size)

    # Test adding transitions
    for i in range(buffer_size * 2):
        state = {
            "audio": torch.randn(state_dim),
            "text": torch.randn(state_dim),
            "emotion_history": torch.zeros(history_length * 5),  # Flattened
            "turn_count": torch.tensor([i % 10], dtype=torch.float32)
        }
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_state = {
            "audio": torch.randn(state_dim),
            "text": torch.randn(state_dim),
            "emotion_history": torch.zeros(history_length * 5),  # Flattened
            "turn_count": torch.tensor([(i + 1) % 10], dtype=torch.float32)
        }
        done = np.random.random() > 0.9  # 10% chance of done

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        buffer.push(transition)

    # Test sampling
    batch_size = 32
    batch = buffer.sample(batch_size)
    
    assert len(batch) == batch_size
    assert all(isinstance(t, Transition) for t in batch)
    
    # Test buffer size doesn't exceed capacity
    assert len(buffer) == buffer_size
