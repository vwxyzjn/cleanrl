import gym
import torch

from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import create_gym_env

dummy_value = torch.ones(1, device="cpu")
envs = create_gym_env("halfcheetah", batch_size=2)
envs.is_vector_env = True
envs = JaxToTorchWrapper(envs, device="cpu")
envs = gym.wrappers.RecordVideo(envs, f"videos/halfcheetah", video_length=50)

obs = envs.reset()
for _ in range(100):
    obs, reward, done, info = envs.step(envs.action_space.sample())
    if done[0]:
        break

