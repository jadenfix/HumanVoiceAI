"""Local DQN optimizer workflow over the explicitly simulated environment.

The environment currently emits placeholder observations/rewards. This test
checks replay, optimizer updates and persistence, not live speech or policy quality.
"""
import numpy as np
import torch
from unittest.mock import MagicMock

from human_voice_ai.policy.rl_policy import EmotionPolicy
from human_voice_ai.rl.environment import VoiceAIEnvironment


def test_local_replay_optimizer_checkpoint_workflow(tmp_path):
    env = VoiceAIEnvironment(ser_model=MagicMock(), tts_engine=MagicMock(),
                             max_turns=6, state_dim=16, history_length=5)
    policy = EmotionPolicy(state_dim=16, action_dim=5, batch_size=4,
                           hidden_dim=16, device="cpu")
    before = [value.detach().clone() for value in policy.policy_net.parameters()]
    losses = []
    for _ in range(2):
        state = env.reset()
        for _ in range(6):
            features = np.asarray(state["audio"], dtype=np.float32)
            action = policy.select_action(features)
            next_state, reward, done, _ = env.step(action)
            next_features = None if done else np.asarray(next_state["audio"], dtype=np.float32)
            policy.store_transition(features, action, reward, next_features, done)
            loss = policy.update()
            if loss is not None:
                assert np.isfinite(loss)
                losses.append(loss)
            state = next_state
    assert len(policy.memory) == 12
    assert len(losses) == policy.steps_done == 9
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.policy_net.parameters()))
    checkpoint = str(tmp_path / "policy.pt")
    policy.save(checkpoint)
    restored = EmotionPolicy(state_dim=16, action_dim=5, hidden_dim=16, device="cpu")
    restored.load(checkpoint)
    assert restored.steps_done == 9
    for expected, actual in zip(policy.policy_net.parameters(), restored.policy_net.parameters()):
        assert torch.equal(expected, actual)
    assert 0 <= restored.select_action(np.zeros(16, dtype=np.float32), training=False) < 5
