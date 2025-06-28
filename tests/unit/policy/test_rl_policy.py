import pytest
import numpy as np
import torch
from human_voice_ai.policy.rl_policy import DQN, ReplayBuffer, EmotionPolicy, Transition


@pytest.fixture
def sample_transition():
    """Create a sample transition for testing."""
    state = np.random.randn(6).astype(np.float32)
    action = 1
    reward = 1.0
    next_state = np.random.randn(6).astype(np.float32)
    done = False
    return Transition(state, action, reward, next_state, done)


def test_dqn_initialization():
    """Test DQN initialization and forward pass."""
    input_dim = 6
    num_actions = 4
    
    # Test with default parameters
    model = DQN(input_dim, num_actions)
    assert len(list(model.parameters())) > 0  # Should have trainable parameters
    
    # Test input/output dimensions
    test_input = torch.randn(1, input_dim)
    output = model(test_input)
    assert output.shape == (1, num_actions)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test with batch
    batch_size = 32
    test_batch = torch.randn(batch_size, input_dim)
    output_batch = model(test_batch)
    assert output_batch.shape == (batch_size, num_actions)
    
    # Test with different hidden dimensions
    hidden_dims = [32, 64, 128]
    for hdim in hidden_dims:
        model = DQN(input_dim, num_actions, hidden_dims=[hdim])
        output = model(test_input)
        assert output.shape == (1, num_actions)


def test_dqn_device_handling():
    """Test DQN device handling."""
    input_dim = 6
    num_actions = 4
    
    # Test CPU
    model_cpu = DQN(input_dim, num_actions)
    assert next(model_cpu.parameters()).device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = DQN(input_dim, num_actions).cuda()
        assert next(model_cuda.parameters()).device.type == 'cuda'


def test_replay_buffer(sample_transition):
    """Test replay buffer functionality."""
    # Test initialization
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0
    
    # Test adding single transition
    buffer.push(sample_transition)
    assert len(buffer) == 1
    
    # Test buffer capacity
    for _ in range(15):  # More than capacity
        buffer.push(sample_transition)
    assert len(buffer) == 10  # Should not exceed capacity

    # Test sampling
    batch_size = 5
    batch = buffer.sample(batch_size)
    assert len(batch) == 5
    assert all(isinstance(t, Transition) for t in batch)
    
    # Test sampling with batch size larger than buffer
    large_batch = buffer.sample(20)
    assert len(large_batch) == 10  # Should return at most buffer size
    
    # Test empty buffer - the current implementation doesn't raise an error when empty
    empty_buffer = ReplayBuffer(capacity=10)
    assert len(empty_buffer.sample(1)) == 0  # Returns empty list when empty


def test_emotion_policy_initialization():
    """Test EmotionPolicy initialization and basic functionality."""
    state_dim = 6
    num_actions = 4
    policy = EmotionPolicy(state_dim, num_actions)

    # Check device
    assert policy.device in ["mps", "cpu"]

    # Check networks
    assert hasattr(policy, "policy_net")
    assert hasattr(policy, "target_net")
    assert hasattr(policy, "memory")
