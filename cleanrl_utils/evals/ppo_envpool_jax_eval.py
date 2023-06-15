import os
from typing import Callable

import cv2
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    seed=1,
):
    envs = make_env(env_id, seed, num_envs=1)()
    Network, Actor, Critic = Model
    next_obs = envs.reset()
    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    key = jax.random.PRNGKey(seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    network_params = network.init(network_key, np.array([envs.single_observation_space.sample()]))
    actor_params = actor.init(actor_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
    critic_params = critic.init(critic_key, network.apply(network_params, np.array([envs.single_observation_space.sample()])))
    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (args, (network_params, actor_params, critic_params)) = flax.serialization.from_bytes(
            (None, (network_params, actor_params, critic_params)), f.read()
        )

    @jax.jit
    def get_action_and_value(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        hidden = network.apply(network_params, next_obs)
        logits = actor.apply(actor_params, hidden)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    # a simple non-vectorized version

    episodic_returns = []
    for episode in range(eval_episodes):
        episodic_return = 0
        next_obs = envs.reset()
        terminated = False

        if capture_video:
            recorded_frames = []
            # conversion from grayscale into rgb
            recorded_frames.append(cv2.cvtColor(next_obs[0][-1], cv2.COLOR_GRAY2RGB))
        while not terminated:
            actions, key = get_action_and_value(network_params, actor_params, next_obs, key)
            next_obs, _, _, infos = envs.step(np.array(actions))
            episodic_return += infos["reward"][0]
            terminated = sum(infos["terminated"]) == 1

            if capture_video and episode == 0:
                recorded_frames.append(cv2.cvtColor(next_obs[0][-1], cv2.COLOR_GRAY2RGB))

            if terminated:
                print(f"eval_episode={len(episodic_returns)}, episodic_return={episodic_return}")
                episodic_returns.append(episodic_return)
                if capture_video and episode == 0:
                    clip = ImageSequenceClip(recorded_frames, fps=24)
                    os.makedirs(f"videos/{run_name}", exist_ok=True)
                    clip.write_videofile(f"videos/{run_name}/{episode}.mp4", logger="bar")

    return episodic_returns


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    from cleanrl.ppo_atari_envpool_xla_jax_scan import Actor, Critic, Network, make_env

    model_path = hf_hub_download(
        repo_id="vwxyzjn/Pong-v5-ppo_atari_envpool_xla_jax_scan-seed1", filename="ppo_atari_envpool_xla_jax_scan.cleanrl_model"
    )
    evaluate(
        model_path,
        make_env,
        "Pong-v5",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Network, Actor, Critic),
        capture_video=False,
    )
