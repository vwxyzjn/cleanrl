import random
from argparse import Namespace
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    model = Model(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max)
    model.load_state_dict(model_data["model_weights"])
    model = model.to(device)
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = model.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()
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

    from cleanrl.c51 import QNetwork, make_env

    model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-c51-seed1", filename="c51.cleanrl_model")
    evaluate(
        model_path,
        make_env,
        "CartPole-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False,
    )
