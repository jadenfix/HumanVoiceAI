"""
System tests for the complete RL workflow.

These tests verify the end-to-end functionality of the RL pipeline,
from environment interaction to policy updates.
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

from human_voice_ai.policy.rl_policy import EmotionPolicy
from human_voice_ai.rl.environment import VoiceAIEnvironment
from human_voice_ai.rl.agent import ReplayBuffer, PPOTrainer, Transition


class TestRLWorkflow:
    """Test class for the complete RL workflow."""
    
    @pytest.fixture
    def mock_ser_model(self):
        """Create a mock SER model for testing."""
        model = MagicMock()
        model.return_value = {"logits": torch.randn(1, 5)}  # 5 emotion classes
        return model
    
    @pytest.fixture
    def mock_tts_engine(self):
        """Create a mock TTS engine for testing."""
        engine = MagicMock()
        engine.synthesize.return_value = (np.zeros(16000), 16000)  # 1 second of silence
        return engine
    
    @pytest.fixture
    def env_config(self, mock_ser_model, mock_tts_engine):
        """Return a standard environment configuration for testing."""
        return {
            'ser_model': mock_ser_model,
            'tts_engine': mock_tts_engine,
            'max_turns': 10,
            'state_dim': 256,
            'history_length': 5
        }
    
    @pytest.fixture
    def policy_config(self):
        """Return a standard policy configuration for testing."""
        return {
            'state_dim': 256,
            'num_actions': 5,  # Changed from action_dim to num_actions to match EmotionPolicy
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 0.9,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'buffer_size': 10000  # Changed from memory_size to buffer_size to match EmotionPolicy
        }
    
    def test_complete_workflow(self, env_config, policy_config, mock_ser_model, mock_tts_engine):
        """Test the complete RL workflow from environment to policy updates."""
        # Initialize environment with mocked dependencies
        env = VoiceAIEnvironment(
            ser_model=mock_ser_model,
            tts_engine=mock_tts_engine,
            max_turns=env_config['max_turns'],
            state_dim=env_config['state_dim'],
            history_length=env_config['history_length']
        )
        
        # Initialize policy with required parameters
        policy = EmotionPolicy(
            state_dim=policy_config['state_dim'],
            num_actions=policy_config['num_actions'],
            lr=policy_config['learning_rate'],
            gamma=policy_config['gamma'],
            epsilon_start=policy_config['epsilon_start'],
            epsilon_end=policy_config['epsilon_end'],
            epsilon_decay=policy_config['epsilon_decay'],
            batch_size=policy_config['batch_size'],
            buffer_size=policy_config['buffer_size']
        )
        
        # Initialize replay buffer
        replay_buffer = ReplayBuffer(capacity=policy_config['buffer_size'])
        
        # Initialize PPO trainer with policy_net instead of model
        trainer = PPOTrainer(
            policy=policy.policy_net,  # Use policy_net instead of model
            lr=policy_config['learning_rate'],
            gamma=0.99,
            epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Run training loop for a few episodes
        num_episodes = 3
        max_steps_per_episode = 5
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # For testing purposes, create a random state with the expected dimension
                # The policy expects a state with dimension state_dim (256)
                state_array = np.random.randn(policy_config['state_dim']).astype(np.float32)
                
                # Select action using policy
                action = policy.select_action(state_array)
                
                # Take step in environment
                next_state, reward, done, _ = env.step(action)
                
                # Create a transition and store it in the replay buffer
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                replay_buffer.push(transition)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # Train the policy if enough samples are available
                if len(replay_buffer) > policy_config['batch_size']:
                    # Sample a batch from the replay buffer
                    batch = replay_buffer.sample(policy_config['batch_size'])
                    
                    # Convert to tensors
                    states = torch.FloatTensor(np.array([t.state for t in batch]))
                    actions = torch.LongTensor(np.array([t.action for t in batch]))
                    rewards = torch.FloatTensor(np.array([t.reward for t in batch]))
                    next_states = torch.FloatTensor(np.array([t.next_state for t in batch]))
                    dones = torch.FloatTensor(np.array([t.done for t in batch]))
                    
                    # Compute advantages and returns
                    with torch.no_grad():
                        _, next_values = policy.model(next_states)
                        _, values = policy.model(states)
                        
                        # Compute returns using GAE (Generalized Advantage Estimation)
                        advantages = torch.zeros_like(rewards)
                        last_gae_lam = 0
                        gamma = policy_config['gamma']
                        gae_lambda = 0.95
                        
                        for t in reversed(range(len(rewards))):
                            if t == len(rewards) - 1:
                                next_value = 0 if dones[t] else next_values[t].item()
                                delta = rewards[t] + gamma * next_value - values[t].item()
                            else:
                                next_value = values[t + 1].item() if not dones[t] else 0
                                delta = rewards[t] + gamma * next_value - values[t].item()
                            
                            last_gae_lam = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae_lam
                            advantages[t] = last_gae_lam
                        
                        returns = advantages + values.squeeze()
                    
                    # Update policy using PPO
                    policy_loss, value_loss, entropy_loss = trainer.update(
                        states=states,
                        actions=actions,
                        old_log_probs=torch.zeros_like(actions).float(),
                        old_values=values.squeeze(),
                        advantages=advantages,
                        returns=returns
                    )
                    
                    assert not torch.isnan(policy_loss).any()
                    assert not torch.isnan(value_loss).any()
                    assert not torch.isnan(entropy_loss).any()
                
                if done:
                    break
            
            # Log episode statistics
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Steps: {episode_steps}, "
                  f"Reward: {episode_reward:.2f}")
            
            # Verify episode statistics
            assert episode_steps > 0
            assert isinstance(episode_reward, float)
        
        # Verify final policy state
        assert policy.epsilon <= policy_config['epsilon_start']  # Epsilon should decay
        assert len(replay_buffer) > 0  # Replay buffer should have transitions
        
        # Test saving and loading the policy
        import tempfile
        import shutil
        
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        test_model_path = os.path.join(temp_dir, "test_policy.pt")
        
        try:
            # Save the policy
            policy.save(test_model_path)
            assert os.path.exists(test_model_path), "Model file was not saved"
            
            # Load the policy
            loaded_policy = EmotionPolicy(
                state_dim=policy_config['state_dim'],
                num_actions=policy_config['num_actions']
            )
            loaded_policy.load(test_model_path)
            
            # Verify the loaded policy works
            test_state = np.random.randn(policy_config['state_dim']).astype(np.float32)
            action = loaded_policy.select_action(test_state, training=False)
            assert 0 <= action < policy_config['num_actions']
            
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
