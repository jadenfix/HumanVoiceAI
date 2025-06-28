"""
Reinforcement Learning Agent for HumanVoiceAI.

This module implements the RL agent using PPO for emotion selection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class Transition:
    """Container for a single transition in the replay buffer."""
    state: Dict[str, torch.Tensor]
    action: int
    reward: float
    next_state: Dict[str, torch.Tensor]
    done: bool

class ReplayBuffer:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)

class PolicyNetwork(nn.Module):
    """Neural network policy for emotion selection."""
    
    def __init__(self, 
                 state_dim: int = 128, 
                 hidden_dim: int = 256,
                 num_actions: int = 5):
        """Initialize the policy network.
        
        Args:
            state_dim: Dimension of state representations
            hidden_dim: Size of hidden layers
            num_actions: Number of possible actions (emotions)
        """
        super().__init__()
        
        # Process different parts of the state
        self.audio_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        self.text_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + 5 + 1, hidden_dim),  # +5 for emotion history, +1 for turn count
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: Dictionary containing state tensors
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        # Process audio features
        audio_features = self.audio_net(state['audio'])
        
        # Process text features
        text_features = self.text_net(state['text'])
        
        # Combine features
        combined = torch.cat([
            audio_features,
            text_features,
            state['emotion_history'].view(state['emotion_history'].size(0), -1),
            state['turn_count']
        ], dim=1)
        
        # Process combined features
        features = self.combined_net(combined)
        
        # Get action logits and state value
        action_logits = self.actor(features)
        state_value = self.critic(features).squeeze(-1)
        
        return action_logits, state_value
    
    def act(self, state: Dict[str, torch.Tensor], epsilon: float = 0.1) -> int:
        """Select an action using an epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.actor.out_features)
            
        with torch.no_grad():
            action_logits, _ = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs).item()
            
        return action

class PPOTrainer:
    """PPO implementation for training the policy."""
    
    def __init__(self,
                 policy: PolicyNetwork,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5):
        """Initialize the PPO trainer.
        
        Args:
            policy: Policy network
            lr: Learning rate
            gamma: Discount factor
            epsilon: Clip parameter for PPO
            entropy_coef: Weight for entropy bonus
            value_coef: Weight for value loss
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, batch: List[Transition]) -> Dict[str, float]:
        """Update the policy using PPO.
        
        Args:
            batch: List of transitions
            
        Returns:
            Dictionary of training metrics
        """
        # Convert batch to tensors
        states = {
            'audio': torch.stack([t.state['audio'] for t in batch]),
            'text': torch.stack([t.state['text'] for t in batch]),
            'emotion_history': torch.stack([t.state['emotion_history'] for t in batch]),
            'turn_count': torch.stack([t.state['turn_count'] for t in batch])
        }
        
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)
        
        # Get old action probabilities
        with torch.no_grad():
            old_logits, _ = self.policy(states)
            old_probs = F.softmax(old_logits, dim=-1)
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate advantages and returns
        with torch.no_grad():
            _, values = self.policy(states)
            next_values = torch.cat([values[1:], torch.zeros(1)])
            next_values[dones.bool()] = 0
            
            deltas = rewards + self.gamma * next_values - values
            advantages = torch.zeros_like(deltas)
            advantages[-1] = deltas[-1]
            
            for t in reversed(range(len(deltas) - 1)):
                advantages[t] = deltas[t] + self.gamma * advantages[t + 1]
            
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(4):  # Number of PPO epochs
            # Get current action probabilities
            logits, values_pred = self.policy(states)
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Calculate ratio
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            ratio = (action_log_probs - old_action_log_probs.detach()).exp()
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_pred, returns)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, values = self.policy(states)
            value_loss = F.mse_loss(values, returns)
            
            # Calculate explained variance
            y_true = returns.numpy()
            y_pred = values.numpy()
            explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true)
            
        return {
            'loss': loss.item(),
            'value_loss': value_loss.item(),
            'explained_variance': explained_var,
            'entropy': entropy.item(),
            'avg_reward': rewards.mean().item()
        }
