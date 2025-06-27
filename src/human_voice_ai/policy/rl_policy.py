"""
Reinforcement learning policy for emotion-aware responses.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional, Dict, Any
import os

class EmotionPolicyNetwork(nn.Module):
    """Neural network for RL policy."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        """Initialize the policy network.
        
        Args:
            input_dim: Dimension of the input state
            hidden_dim: Number of hidden units
            num_actions: Number of possible actions
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class RLPolicy:
    """Reinforcement learning policy for emotion-aware responses."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update: int = 10,
        device: str = "cpu"
    ):
        """Initialize the RL policy.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            hidden_dim: Number of hidden units in the network
            lr: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate of epsilon decay
            batch_size: Training batch size
            buffer_size: Size of the replay buffer
            target_update: Update target network every N steps
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.steps_done = 0
        
        # Initialize networks
        self.policy_net = EmotionPolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net = EmotionPolicyNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            eval_mode: If True, disable exploration
            
        Returns:
            Selected action
        """
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """Update the policy network using a batch of transitions.
        
        Returns:
            Loss value if an update was performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s_t, a)
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            expected_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path: str) -> None:
        """Save the policy to a file.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.policy_net.net[0].out_features,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update': self.target_update
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'RLPolicy':
        """Load a policy from a file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded RLPolicy instance
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        policy = cls(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['hidden_dim'],
            lr=1e-4,  # Will be loaded from optimizer state
            gamma=config['gamma'],
            epsilon=config['epsilon'],
            epsilon_min=config['epsilon_min'],
            epsilon_decay=config['epsilon_decay'],
            batch_size=config['batch_size'],
            target_update=config['target_update'],
            device=device
        )
        
        policy.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        policy.target_net.load_state_dict(checkpoint['target_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        policy.epsilon = checkpoint['epsilon']
        policy.steps_done = checkpoint.get('steps_done', 0)
        
        return policy


def test_policy():
    """Test function for the RL policy."""
    print("Testing RL policy...")
    
    # Create a simple environment
    state_dim = 8  # Number of emotion classes
    action_dim = 5  # Number of possible actions
    
    # Initialize policy
    policy = RLPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        batch_size=32,
        buffer_size=1000,
        device="cpu"
    )
    
    # Test action selection
    state = np.random.rand(state_dim)
    action = policy.select_action(state)
    print(f"Selected action: {action}")
    
    # Test training loop
    for _ in range(10):
        # Simulate environment
        next_state = np.random.rand(state_dim)
        reward = np.random.uniform(-1, 1)
        done = np.random.random() < 0.1
        
        # Store transition
        policy.memory.push(state, action, reward, next_state, done)
        
        # Update policy
        loss = policy.update()
        if loss is not None:
            print(f"Loss: {loss:.4f}, Epsilon: {policy.epsilon:.4f}")
        
        state = next_state
    
    print("Test completed")


if __name__ == "__main__":
    test_policy()
