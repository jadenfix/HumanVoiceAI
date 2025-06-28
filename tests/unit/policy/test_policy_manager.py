import pytest
import numpy as np
from pathlib import Path
from human_voice_ai.policy.manager import PolicyManager, Action

def test_policy_manager_initialization(test_config_path):
    """Test PolicyManager initialization."""
    manager = PolicyManager(config_path=test_config_path)
    
    # Check if config is loaded
    assert hasattr(manager, 'config')
    assert 'actions' in manager.config
    assert len(manager.actions) > 0
    
    # Check action space
    assert all(isinstance(action, Action) for action in manager.actions)
    
    # Check policy initialization
    assert hasattr(manager, 'policy')
    assert manager.policy is not None

def test_action_selection(test_config_path):
    """Test action selection with PolicyManager."""
    manager = PolicyManager(config_path=test_config_path)
    
    # Test with emotion probabilities
    emotion_probs = [0.1, 0.7, 0.05, 0.05, 0.05, 0.05]  # Mostly "happy"
    action = manager.get_action(emotion_probs)
    
    # Check action properties
    assert hasattr(action, 'id')
    assert hasattr(action, 'name')
    assert hasattr(action, 'description')
    assert 0 <= action.id < len(manager.actions)

def test_get_available_actions(test_config_path):
    """Test getting available actions."""
    manager = PolicyManager(config_path=test_config_path)
    actions = manager.get_available_actions()
    
    # Check actions
    assert isinstance(actions, list)
    assert len(actions) > 0
    assert all(isinstance(action, Action) for action in actions)
