import random
from typing import Callable

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import numpy as np


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    epsilon: float = 0.05,
    capture_video: bool = True,
    seed=1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    obs, _ = envs.reset()
    model = Model(action_dim=envs.single_action_space.n)
    q_key = jax.random.PRNGKey(seed)
    params = model.init(q_key, obs)
    with open(model_path, "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())
    model.apply = jax.jit(model.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model.apply(params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.dqn_jax import QNetwork, make_env

    model_path = hf_hub_download(repo_id="vwxyzjn/CartPole-v1-dqn_jax-seed1", filename="dqn_jax.cleanrl_model")
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        capture_video=False,
    )
