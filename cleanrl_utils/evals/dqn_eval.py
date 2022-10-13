import random
from typing import Callable

import torch
import gym
import numpy as np

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    QNetwork: torch.nn.Module,
    device: torch.device,
    epsilon: float = 0.05,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, True, run_name)])
    q_network = QNetwork(envs).to(device)
    q_network.load_state_dict(torch.load(model_path))
    q_network.eval()

    obs = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, infos = envs.step(actions)
        for info in infos:
            if "episode" in info.keys():
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns