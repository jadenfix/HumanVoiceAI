import pytest
import numpy as np
import torch
from human_voice_ai.policy.rl_policy import DQN, ReplayBuffer, EmotionPolicy

def test_dqn_initialization():
    """Test DQN initialization and forward pass."""
    input_dim = 6
    num_actions = 4
    model = DQN(input_dim, num_actions)
    
    # Test input/output dimensions
    test_input = torch.randn(1, input_dim)
    output = model(test_input)
    assert output.shape == (1, num_actions)
    
    # Test with batch
    batch_size = 32
    test_batch = torch.randn(batch_size, input_dim)
    output_batch = model(test_batch)
    assert output_batch.shape == (batch_size, num_actions)

def test_replay_buffer(sample_transition):
    """Test replay buffer functionality."""
    buffer = ReplayBuffer(capacity=10)
    
    # Test adding single transition
    buffer.push(sample_transition)
    assert len(buffer) == 1
    
    # Test buffer capacity
    for _ in range(15):  # More than capacity
        buffer.push(sample_transition)
    assert len(buffer) == 10  # Should not exceed capacity
    
    # Test sampling
    batch = buffer.sample(5)
    assert len(batch) == 5
    assert hasattr(batch[0], 'state')
    
    # Test sampling more than buffer size
    large_batch = buffer.sample(20)
    assert len(large_batch) == 10  # Should return at most buffer size

def test_emotion_policy_initialization():
    """Test EmotionPolicy initialization."""
    state_dim = 6
    num_actions = 4
    policy = EmotionPolicy(state_dim, num_actions)
    
    # Check device
    assert policy.device in ["mps", "cpu"]
    
    # Check networks
    assert hasattr(policy, 'policy_net')
    assert hasattr(policy, 'target_net')
    assert hasattr(policy, 'memory')
