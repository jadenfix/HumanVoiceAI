#!/usr/bin/env python3
"""
Training script for the HumanVoiceAI RL agent.

This script trains an RL agent to select appropriate emotions for TTS responses.
"""

import os
import time
import random
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.human_voice_ai.rl.agent import PolicyNetwork, PPOTrainer, ReplayBuffer
from src.human_voice_ai.rl.environment import VoiceAIEnvironment
from src.human_voice_ai.tts.tts_engine import TtsEngine

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RLTrainingConfig:
    """Configuration for RL training."""
    
    def __init__(self):
        # Training parameters
        self.num_episodes = 1000
        self.max_steps_per_episode = 20
        self.batch_size = 64
        self.buffer_size = 10000
        self.learning_starts = 1000
        self.train_freq = 4
        self.target_update_freq = 100
        
        # PPO parameters
        self.gamma = 0.99
        self.epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.lr = 3e-4
        
        # Model architecture
        self.state_dim = 128
        self.hidden_dim = 256
        
        # Exploration
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.01
        self.epsilon_decay = 0.995
        
        # Logging and saving
        self.log_dir = "logs/rl_training"
        self.save_dir = "saved_models"
        self.save_freq = 100
        
        # Environment
        self.history_length = 5

def create_environment() -> VoiceAIEnvironment:
    """Create and return the VoiceAI environment."""
    # TODO: Initialize actual SER model and TTS engine
    ser_model = None
    tts_engine = TtsEngine()
    
    env = VoiceAIEnvironment(
        ser_model=ser_model,
        tts_engine=tts_engine,
        max_turns=20,
        state_dim=128,
        history_length=5
    )
    
    return env

def evaluate_agent(agent: PolicyNetwork, 
                  env: VoiceAIEnvironment, 
                  num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate the agent's performance.
    
    Args:
        agent: The policy network to evaluate
        env: The environment to evaluate in
        num_episodes: Number of episodes to run
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < env.max_turns:
            # Convert state to tensor
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    state_tensor[k] = torch.FloatTensor(v).unsqueeze(0)
                else:
                    state_tensor[k] = torch.FloatTensor([v]).unsqueeze(0)
            
            # Select action
            with torch.no_grad():
                action = agent.act(state_tensor, epsilon=0.0)  # No exploration
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
    
    return {
        'mean_reward': float(np.mean(total_rewards)),
        'std_reward': float(np.std(total_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'success_rate': float(np.mean([1 if r > 0 else 0 for r in total_rewards]))
    }

def train() -> None:
    """Main training loop."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL agent for HumanVoiceAI')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize config
    config = RLTrainingConfig()
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    
    # Initialize logging
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # Create environment
    env = create_environment()
    
    # Initialize agent and replay buffer
    policy = PolicyNetwork(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        num_actions=5  # Number of emotions
    ).to(args.device)
    
    trainer = PPOTrainer(
        policy=policy,
        lr=config.lr,
        gamma=config.gamma,
        epsilon=config.epsilon,
        entropy_coef=config.entropy_coef,
        value_coef=config.value_coef,
        max_grad_norm=config.max_grad_norm
    )
    
    replay_buffer = ReplayBuffer(capacity=config.buffer_size)
    
    # Training loop
    global_step = 0
    epsilon = config.initial_epsilon
    
    for episode in range(1, config.num_episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config.max_steps_per_episode:
            # Convert state to tensor
            state_tensor = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    state_tensor[k] = torch.FloatTensor(v).unsqueeze(0).to(args.device)
                else:
                    state_tensor[k] = torch.FloatTensor([v]).unsqueeze(0).to(args.device)
            
            # Select action
            action = trainer.policy.act(state_tensor, epsilon=epsilon)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            replay_buffer.push(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            ))
            
            # Update state and counters
            state = next_state
            episode_reward += reward
            step += 1
            global_step += 1
            
            # Train the agent
            if len(replay_buffer) >= config.learning_starts and global_step % config.train_freq == 0:
                batch = replay_buffer.sample(config.batch_size)
                metrics = trainer.update(batch)
                
                # Log metrics
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step)
            
            # Update exploration rate
            epsilon = max(config.final_epsilon, epsilon * config.epsilon_decay)
        
        # Log episode metrics
        writer.add_scalar('train/episode_reward', episode_reward, episode)
        writer.add_scalar('train/episode_length', step, episode)
        writer.add_scalar('train/epsilon', epsilon, episode)
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode}/{config.num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}")
        
        # Evaluate agent
        if episode % config.save_freq == 0:
            eval_metrics = evaluate_agent(policy, env)
            for k, v in eval_metrics.items():
                writer.add_scalar(f'eval/{k}', v, episode)
            
            print(f"\nEvaluation after episode {episode}:")
            for k, v in eval_metrics.items():
                print(f"  {k}: {v:.3f}")
            print()
            
            # Save model
            model_path = os.path.join(config.save_dir, f'policy_episode_{episode}.pt')
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'episode': episode,
                'eval_metrics': eval_metrics,
                'config': vars(config)
            }, model_path)
            print(f"Model saved to {model_path}")
    
    # Close environment and writer
    env.close()
    writer.close()

if __name__ == "__main__":
    train()
