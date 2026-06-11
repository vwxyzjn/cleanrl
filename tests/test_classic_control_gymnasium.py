import subprocess

import gymnasium as gym
import numpy as np
import torch

from cleanrl.ppo_continuous_action import Agent, make_env
from cleanrl_utils.evals.ppo_eval import evaluate


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51():
    subprocess.run(
        "python cleanrl/c51.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_eval():
    subprocess.run(
        "python cleanrl/c51.py --save-model --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_ppo_continuous_action_eval():
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --save-model --env-id Pendulum-v1 --num-envs 1 --num-steps 64 --total-timesteps 128",
        shell=True,
        check=True,
    )


def test_ppo_continuous_action_eval_restores_obs_rms(tmp_path):
    """The evaluation envs must reuse the observation normalization statistics from training (#532)."""
    envs = gym.vector.SyncVectorEnv([make_env("Pendulum-v1", 0, False, "test", 0.99)])
    agent = Agent(envs)
    obs_rms = envs.envs[0].get_wrapper_attr("obs_rms")
    obs_rms.mean = np.full_like(obs_rms.mean, 5.0)
    obs_rms.var = np.full_like(obs_rms.var, 100.0)
    obs_rms.count = 10000.0
    model_path = f"{tmp_path}/ppo_continuous_action.cleanrl_model"
    model_data = {
        "model_weights": agent.state_dict(),
        "obs_rms": {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count},
    }
    torch.save(model_data, model_path)
    envs.close()

    eval_envs = []

    def make_env_capture(env_id, idx, capture_video, run_name, gamma):
        thunk = make_env(env_id, idx, capture_video, run_name, gamma)

        def wrapped_thunk():
            env = thunk()
            eval_envs.append(env)
            return env

        return wrapped_thunk

    episodic_returns = evaluate(
        model_path,
        make_env_capture,
        "Pendulum-v1",
        eval_episodes=1,
        run_name="test-eval",
        Model=Agent,
        capture_video=False,
    )
    assert len(episodic_returns) == 1
    # the statistics are restored from the checkpoint and stay frozen during evaluation
    eval_obs_rms = eval_envs[0].get_wrapper_attr("obs_rms")
    np.testing.assert_array_equal(eval_obs_rms.mean, obs_rms.mean)
    np.testing.assert_array_equal(eval_obs_rms.var, obs_rms.var)
    assert eval_obs_rms.count == obs_rms.count


def test_ppo_continuous_action_eval_legacy_checkpoint(tmp_path):
    """Checkpoints saved as a plain `state_dict` (e.g. the hub models) must still be loadable."""
    envs = gym.vector.SyncVectorEnv([make_env("Pendulum-v1", 0, False, "test", 0.99)])
    agent = Agent(envs)
    model_path = f"{tmp_path}/ppo_continuous_action.cleanrl_model"
    torch.save(agent.state_dict(), model_path)
    envs.close()

    episodic_returns = evaluate(
        model_path,
        make_env,
        "Pendulum-v1",
        eval_episodes=1,
        run_name="test-eval-legacy",
        Model=Agent,
        capture_video=False,
    )
    assert len(episodic_returns) == 1
