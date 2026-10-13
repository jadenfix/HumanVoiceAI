"""Numeric and mode regressions use real networks and optimizer updates."""
import numpy as np
import pytest
import torch
from human_voice_ai.policy.rl_policy import EmotionPolicy
from human_voice_ai.rl.agent import PolicyNetwork, PPOTrainer, Transition


def test_greedy_actions_disable_dropout_and_restore_model_mode():
    policy = EmotionPolicy(4, 2, hidden_dim=8, epsilon_start=0, device="cpu")
    layers = list(policy.policy_net.network)
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        layers[0].weight[0, 0] = 1
        layers[3].weight[0, 0] = 1
        layers[6].weight[0, 0] = 1
        layers[6].bias[1] = .1
    state = np.ones(4, dtype=np.float32)
    for training_mode in (True, False):
        policy.policy_net.train(training_mode)
        assert {policy.select_action(state, training=False) for _ in range(50)} == {0}
        assert policy.policy_net.training is training_mode


def ppo_fixture():
    policy = PolicyNetwork(state_dim=4, hidden_dim=8, num_actions=5)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    state = {"audio": torch.zeros(4), "text": torch.zeros(4),
             "emotion_history": torch.zeros(5), "turn_count": torch.zeros(1)}
    return policy, state


def test_singleton_ppo_update_keeps_metrics_parameters_and_gradients_finite():
    policy, state = ppo_fixture()
    before = [value.detach().clone() for value in policy.parameters()]
    metrics = PPOTrainer(policy).update([Transition(state, 0, 1., state, True)])
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(torch.isfinite(value).all() for value in policy.parameters())
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in policy.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))


def test_ppo_returns_do_not_cross_terminal_episode_boundaries(monkeypatch):
    policy, state = ppo_fixture()
    targets = []
    real_loss = torch.nn.functional.mse_loss
    def capture(prediction, target, *args, **kwargs):
        assert prediction.shape == target.shape == (2,)
        targets.append(target.detach().clone())
        return real_loss(prediction, target, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", capture)
    metrics = PPOTrainer(policy).update([
        Transition(state, 0, 1., state, True),
        Transition(state, 0, 100., state, True),
    ])
    assert targets[0].tolist() == [1., 100.]
    assert all(np.isfinite(value) for value in metrics.values())
