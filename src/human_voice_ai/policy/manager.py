# src/human_voice_ai/policy/manager.py
from typing import Dict, Any, Optional, List
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from .rl_policy import EmotionPolicy, Transition

@dataclass
class Action:
    """An action that the policy can take."""
    id: int
    name: str
    description: str

class PolicyManager:
    """Manages the RL policy for emotion-based responses."""
    
    def __init__(
        self,
        config_path: str = "configs/rl_policy_config.yaml",
        model_path: Optional[str] = None
    ):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize actions
        self.actions = [
            Action(i, action["name"], action["description"])
            for i, action in enumerate(self.config["actions"])
        ]
        
        # Initialize policy
        state_dim = 6  # Number of emotion classes
        num_actions = len(self.actions)
        
        self.policy = EmotionPolicy(
            state_dim=state_dim,
            num_actions=num_actions,
            lr=self.config["training"]["lr"],
            gamma=self.config["training"]["gamma"],
            epsilon_start=self.config["exploration"]["epsilon_start"],
            epsilon_end=self.config["exploration"]["epsilon_end"],
            epsilon_decay=self.config["exploration"]["epsilon_decay"],
            batch_size=self.config["training"]["batch_size"],
            buffer_size=self.config["training"]["buffer_size"],
            target_update=self.config["training"]["target_update"]
        )
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def get_action(self, emotion_probs: np.ndarray) -> Action:
        """Get an action based on the current emotion probabilities."""
        if len(emotion_probs) != 6:
            raise ValueError("Expected 6 emotion probabilities")
            
        action_idx = self.policy.select_action(np.array(emotion_probs))
        return self.actions[action_idx]
        
    def update_policy(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool
    ) -> Optional[float]:
        """Update the policy with a new transition."""
        # Add transition to replay buffer
        transition = Transition(state, action, reward, next_state, done)
        self.policy.memory.push(transition)
        
        # Update the policy
        return self.policy.update()
        
    def save_model(self, path: str) -> None:
        """Save the policy model to a file."""
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.policy.save(path)
        
    def load_model(self, path: str) -> None:
        """Load the policy model from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        self.policy.load(path)
        
    def get_available_actions(self) -> List[Action]:
        """Get a list of all available actions."""
        return self.actions.copy()
        
    def get_action_by_name(self, name: str) -> Optional[Action]:
        """Get an action by its name."""
        for action in self.actions:
            if action.name == name:
                return action
        return None