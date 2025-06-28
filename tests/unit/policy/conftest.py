import pytest
import numpy as np
import os
from human_voice_ai.policy.rl_policy import Transition
from pathlib import Path

@pytest.fixture
def sample_transition():
    """Create a sample transition for testing."""
    return Transition(
        state=np.random.randn(6),  # 6 emotion classes
        action=0,
        reward=1.0,
        next_state=np.random.randn(6),
        done=False
    )

@pytest.fixture
def test_config_path():
    """Return path to test config file."""
    return str(Path(__file__).parent.parent.parent / "fixtures" / "test_policy_config.yaml")
