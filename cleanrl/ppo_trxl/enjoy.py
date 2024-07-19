import gymnasium as gym
import torch
from ppo_trxl import Agent, make_env

# Load checkpoint
checkpoint = torch.load("./ppo_trxl.cleanrl_model")
args = checkpoint["args"]
args = type("Args", (), args)

# Init env and reset
env = make_env(args.env_id, 0, False, "", "human")()
obs, _ = env.reset()
env.render()

# Determine maximum episode steps
max_episode_steps = env.spec.max_episode_steps
if not max_episode_steps:
    max_episode_steps = env.max_episode_steps
if max_episode_steps <= 0:
    max_episode_steps = 2048  # Memory Gym envs have max_episode_steps set to -1

# Setup agent and load its model parameters
action_space_shape = (
        (env.action_space.n,)
        if isinstance(env.action_space, gym.spaces.Discrete)
        else tuple(env.action_space.nvec)
    )
agent = Agent(args, env.observation_space, action_space_shape, max_episode_steps)
agent.load_state_dict(checkpoint["model_weights"])

# Setup memory, mask and indices
memory = torch.zeros((1, max_episode_steps, args.trxl_num_blocks, args.trxl_dim), dtype=torch.float32)
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

print(info)
env.close()
