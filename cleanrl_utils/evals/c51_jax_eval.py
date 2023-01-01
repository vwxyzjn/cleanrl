import random
from typing import Callable

import flax
import flax.linen as nn
import gym
import jax
import numpy as np
from argparse import Namespace
import jax.numpy as jnp


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
    obs = envs.reset()
    model_data = None
    with open(model_path, "rb") as f:
        model_data = flax.serialization.from_bytes(model_data, f.read())
    args = Namespace(**model_data["args"])
    model = Model(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms)
    # q_key = jax.random.PRNGKey(seed)
    params = model_data["model_weights"]
    model.apply = jax.jit(model.apply)
    atoms = jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms))

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            pmfs = model.apply(params, obs)
            q_vals = (pmfs * atoms).sum(axis=-1)
            actions = q_vals.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
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
