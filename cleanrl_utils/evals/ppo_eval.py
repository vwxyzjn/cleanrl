from typing import Callable

import gymnasium as gym
import torch
import numpy as np


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    gamma: float = 0.99,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, gamma)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            episodes_over = np.nonzero(infos["final_info"]["_episode"])[0]
            episode_returns = infos["final_info"]["episode"]["r"][episodes_over]
            for episode_return in episode_returns:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episode_return}")
                episodic_returns.append(episode_return)
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_continuous_action import Agent, make_env

    model_path = hf_hub_download(
        repo_id="sdpkjc/Hopper-v4-ppo_continuous_action-seed1", filename="ppo_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Hopper-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=Agent,
        device="cpu",
        capture_video=False,
    )
