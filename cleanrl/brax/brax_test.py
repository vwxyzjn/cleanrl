import os
import time
from functools import partial

import gym
import numpy as np
import torch

from brax.envs.to_torch import JaxToTorchWrapper
from brax.envs import _envs, create_gym_env

if 'COLAB_TPU_ADDR' in os.environ:
  from jax.tools import colab_tpu
  colab_tpu.setup_tpu()

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    # BUG: (@lebrice): Getting a weird "CUDA error: out of memory" RuntimeError
    # during JIT, which can be "fixed" by first creating a dummy cuda tensor!
    v = torch.ones(1, device="cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for env_name, env_class in _envs.items():
    env_id = f"brax_{env_name}-v0"
    entry_point = partial(create_gym_env, env_name=env_name)
    if env_id not in gym.envs.registry.env_specs:
        print(f"Registring brax's '{env_name}' env under id '{env_id}'.")
        gym.register(env_id, entry_point=entry_point)


total_timesteps = 0
num_envs = 2048
env = gym.make("brax_halfcheetah-v0", batch_size=num_envs)
env = JaxToTorchWrapper(env)
obs = env.reset()
start_time = time.time()
for _ in range(10000):
    total_timesteps += num_envs
    env.step(env.action_space.sample())


print(f"\nDevice used: {device}")
print(f"Number of parallel environments: {num_envs}")
print(f"FPS: {total_timesteps / (time.time()-start_time)}")
