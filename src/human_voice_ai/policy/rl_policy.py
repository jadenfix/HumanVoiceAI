# src/human_voice_ai/policy/rl_policy.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import os

@dataclass
class Transition:
    """A transition in our environment."""
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.buffer.append(transition)
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for emotion-based policy."""
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert numpy arrays to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
            if torch.cuda.is_available():
                x = x.cuda()
        return self.network(x)
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer)

class DQN(nn.Module):
    """Deep Q-Network for emotion-based policy."""
    
    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class EmotionPolicy:
    """Reinforcement learning policy for emotion-based response selection."""
    
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        device: str = "mps",
        lr: Union[float, str] = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_update: int = 10
    ):
        # Set device - use CUDA if available, then MPS, then CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps_done = 0
        
        # Initialize networks
        self.policy_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net = DQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Convert learning rate to float if it's a string
        if isinstance(lr, str):
            lr = float(lr)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(capacity=buffer_size)
        
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
            
        with torch.no_grad():
            # Convert state to tensor and add batch dimension if needed
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get Q-values and select action with highest Q-value
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def update(self) -> Optional[float]:
        """Perform a single update of the policy network."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*[(t.state, t.action, t.reward, t.next_state, t.done) for t in transitions]))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=self.device, dtype=torch.bool)
        
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor(
                np.array([s for s in batch.next_state if s is not None])
            ).to(self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        self.steps_done += 1
        return loss.item()
    
    def save(self, path: str) -> None:
        """Save the policy to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)
    
    def load(self, path: str) -> None:
        """Load the policy from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        
        # Make sure models are in the right mode
        self.policy_net.train()
        self.target_net.eval()
        
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.policy_net.network[-1].out_features - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
            
    def update(self) -> Optional[float]:
        """Perform a single update of the policy network."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch from replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*[
            (t.state, t.action, t.reward, 
             np.zeros_like(t.state) if t.next_state is None else t.next_state, 
             t.done)
            for t in transitions
        ]))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        # Compute next state values using target network
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool
        )
        
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor(
                np.array([s for s in batch.next_state if s is not None])
            ).to(self.device)
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1)[0].detach()
            )
            
        # Compute expected Q values
        expected_q_values = (next_state_values * self.gamma) + reward_batch
        
        # Get current Q values for the chosen actions
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        self.steps_done += 1
        
        return loss.item()
        
    def save(self, path: str) -> None:
        """Save the policy to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        
    def load(self, path: str) -> None:
        """Load the policy from a file."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.steps_done = checkpoint['steps_done']