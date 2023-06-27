from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    exploration_noise: float = 0.1,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    actor = Model[0](envs).to(device)
    qf = Model[1](envs).to(device)
    actor_params, qf_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    qf.load_state_dict(qf_params)
    qf.eval()
    # note: qf is not used in this script

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions = actor(torch.Tensor(obs).to(device))
            actions += torch.normal(0, actor.action_scale * exploration_noise)
            actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

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

    from cleanrl.ddpg_continuous_action import Actor, QNetwork, make_env

    model_path = hf_hub_download(
        repo_id="cleanrl/HalfCheetah-v4-ddpg_continuous_action-seed1", filename="ddpg_continuous_action.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "HalfCheetah-v4",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_video=False,
    )
