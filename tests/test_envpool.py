import importlib
import runpy
import subprocess
from unittest.mock import Mock

import gym
import numpy as np
import pytest
import torch
import tyro
from torch.utils import tensorboard

ENVPOOL_MODULES = [
    "ppo_atari_envpool",
    "ppo_rnd_envpool",
    "pqn_atari_envpool",
    "pqn_atari_envpool_lstm",
]


class EpisodeBoundaryEnv(gym.Env):
    """Four deterministic streams: timeout, game over, life loss, and continuation."""

    num_envs = 4

    def __init__(self, stack_num=4):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, shape=(stack_num, 84, 84), dtype=np.uint8)
        self.observations = np.zeros((self.num_envs,) + self.observation_space.shape, dtype=np.uint8)
        self.step_index = 0

    def reset(self, **kwargs):
        self.step_index = 0
        return self.observations

    def step(self, action):
        if self.step_index == 0:
            raw_rewards = np.array([2, 3, 4, 5], dtype=np.float32)
            terminated = np.array([0, 1, 0, 0], dtype=np.int32)
            truncated = np.array([True, False, False, False])
            dones = np.array([True, True, True, False])
            lives = np.array([3, 0, 2, 3])
        else:
            raw_rewards = np.array([7, 11, 13, 17], dtype=np.float32)
            terminated = np.ones(self.num_envs, dtype=np.int32)
            truncated = np.zeros(self.num_envs, dtype=bool)
            dones = np.ones(self.num_envs, dtype=bool)
            lives = np.zeros(self.num_envs, dtype=int)
        self.step_index += 1
        return (
            self.observations,
            np.clip(raw_rewards, -1, 1),
            dones,
            {
                "reward": raw_rewards,
                "terminated": terminated,
                "TimeLimit.truncated": truncated,
                "lives": lives,
            },
        )


@pytest.mark.parametrize("module_name", ENVPOOL_MODULES)
def test_record_episode_statistics_boundaries(module_name):
    module = importlib.import_module(f"cleanrl.{module_name}")
    env = module.RecordEpisodeStatistics(EpisodeBoundaryEnv())
    env.reset()
    observations, rewards, dones, info = env.step(np.zeros(4, dtype=int))
    assert observations is env.unwrapped.observations
    np.testing.assert_array_equal(rewards, [1, 1, 1, 1])
    np.testing.assert_array_equal(dones, [True, True, True, False])
    np.testing.assert_array_equal(info["r"], [2, 3, 4, 5])
    np.testing.assert_array_equal(info["l"], [1, 1, 1, 1])

    _, rewards, dones, info = env.step(np.zeros(4, dtype=int))
    np.testing.assert_array_equal(rewards, [1, 1, 1, 1])
    np.testing.assert_array_equal(dones, [True, True, True, True])
    # Only the timeout and true termination start new episodes; life loss does not.
    np.testing.assert_array_equal(info["r"], [7, 11, 17, 22])
    np.testing.assert_array_equal(info["l"], [1, 1, 2, 2])
    np.testing.assert_array_equal(env.episode_returns, [0, 0, 0, 0])
    np.testing.assert_array_equal(env.episode_lengths, [0, 0, 0, 0])
    env.reset()
    _, _, _, info = env.step(np.zeros(4, dtype=int))
    np.testing.assert_array_equal(info["r"], [2, 3, 4, 5])
    np.testing.assert_array_equal(info["l"], [1, 1, 1, 1])


@pytest.mark.parametrize("module_name", ENVPOOL_MODULES)
def test_envpool_logs_episode_boundaries(module_name, monkeypatch):
    import envpool

    def make_env(*args, **kwargs):
        return EpisodeBoundaryEnv(stack_num=kwargs.get("stack_num", 4))

    def parse_args(args_class):
        args = args_class(num_envs=4, num_steps=2, total_timesteps=8, num_minibatches=2, update_epochs=1, cuda=False)
        if hasattr(args, "num_iterations_obs_norm_init"):
            args.num_iterations_obs_norm_init = 0
        return args

    writer = Mock()
    monkeypatch.setattr(envpool, "make", make_env)
    monkeypatch.setattr(tyro, "cli", parse_args)
    monkeypatch.setattr(tensorboard, "SummaryWriter", Mock(return_value=writer))
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        module = importlib.import_module(f"cleanrl.{module_name}")
        runpy.run_path(module.__file__, run_name="__main__")
    finally:
        torch.set_num_threads(num_threads)

    returns = [call.args[1] for call in writer.add_scalar.call_args_list if call.args[0] == "charts/episodic_return"]
    lengths = [call.args[1] for call in writer.add_scalar.call_args_list if call.args[0] == "charts/episodic_length"]
    assert returns == [2, 3, 7, 11, 17, 22]
    assert lengths == [1, 1, 1, 1, 2, 2]


def test_ppo_atari_envpool():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool.py --num-envs 8 --num-steps 32 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_rnd_envpool():
    subprocess.run(
        "python cleanrl/ppo_rnd_envpool.py --num-envs 8 --num-steps 32 --num-iterations-obs-norm-init 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan_eval():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --save-model --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )
