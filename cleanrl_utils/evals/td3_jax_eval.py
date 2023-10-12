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
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    seed=1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    max_action = float(envs.single_action_space.high[0])
    obs, _ = envs.reset()

    Actor, QNetwork = Model
    action_scale = np.array((envs.action_space.high - envs.action_space.low) / 2.0)
    action_bias = np.array((envs.action_space.high + envs.action_space.low) / 2.0)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=action_scale,
        action_bias=action_bias,
    )
    qf = QNetwork()
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)
    actor_params = actor.init(actor_key, obs)
    qf1_params = qf.init(qf1_key, obs, envs.action_space.sample())
    qf2_params = qf.init(qf2_key, obs, envs.action_space.sample())
    with open(model_path, "rb") as f:
        (actor_params, qf1_params, qf2_params) = flax.serialization.from_bytes(
            (actor_params, qf1_params, qf2_params), f.read()
        )
    # note: qf1_params and qf2_params are not used in this script
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = actor.apply(actor_params, obs)
        actions = np.array(
            [
                (
                    jax.device_get(actions)[0]
                    + np.random.normal(0, max_action * exploration_noise, size=envs.single_action_space.shape)
                ).clip(envs.single_action_space.low, envs.single_action_space.high)
            ]
        )

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

    from cleanrl.td3_continuous_action_jax import Actor, QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/HalfCheetah-v4-td3_continuous_action_jax-seed1", filename="td3_continuous_action_jax.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "HalfCheetah-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        exploration_noise=0.1,
    )
