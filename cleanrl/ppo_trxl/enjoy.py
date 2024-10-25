from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro
from ppo_trxl import Agent, make_env


@dataclass
class Args:
    hub: bool = False
    """whether to load the model from the huggingface hub or from the local disk"""
    name: str = "Endless-MortarMayhem-v0_12.nn"
    """path to the model file"""


if __name__ == "__main__":
    # Parse command line arguments and retrieve model path
    cli_args = tyro.cli(Args)
    if cli_args.hub:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id="LilHairdy/cleanrl_memory_gym", filename=cli_args.name)
        except:
            raise RuntimeError(
                "Cannot load model from the huggingface hub. Please install the huggingface_hub pypi package and verify the model name. You can also download the model from the hub manually and load it from disk."
            )
    else:
        path = cli_args.name

    # Load the pre-trained model and the original args used to train it
    checkpoint = torch.load(path)
    args = checkpoint["args"]
    args = type("Args", (), args)

    # Init environment and reset
    env = make_env(args.env_id, 0, False, "", "human")()
    obs, _ = env.reset()
    env.render()

    # Determine maximum episode steps
    max_episode_steps = env.spec.max_episode_steps
    if not max_episode_steps:
        max_episode_steps = env.max_episode_steps
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
        # May episode impacts positional encoding, so make sure to set this accordingly

    # Setup agent and load its model parameters
    action_space_shape = (
        (env.action_space.n,) if isinstance(env.action_space, gym.spaces.Discrete) else tuple(env.action_space.nvec)
    )
    agent = Agent(args, env.observation_space, action_space_shape, max_episode_steps)
    agent.load_state_dict(checkpoint["model_weights"])

    # Setup Transformer-XL memory, mask and indices
    memory = torch.zeros((1, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    repetitions = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long()
    memory_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()
    memory_indices = torch.cat((repetitions, memory_indices))

    # Run episode
    done = False
    t = 0
    while not done:
        # Prepare observation and memory
        obs = torch.Tensor(obs).unsqueeze(0)
        memory_window = memory[0, memory_indices[t].unsqueeze(0)]
        t_ = max(0, min(t, args.trxl_memory_length - 1))
        mask = memory_mask[t_].unsqueeze(0)
        indices = memory_indices[t].unsqueeze(0)
        # Forward agent
        action, _, _, _, new_memory = agent.get_action_and_value(obs, memory_window, mask, indices)
        memory[:, t] = new_memory
        # Step
        obs, reward, termination, truncation, info = env.step(action.cpu().squeeze().numpy())
        env.render()
        done = termination or truncation
        t += 1

    if "r" in info["episode"].keys():
        print(f"Episode return: {info['episode']['r'][0]}, Episode length: {info['episode']['l'][0]}")
    else:
        print(f"Episode return: {info['reward']}, Episode length: {info['length']}")
    env.close()
