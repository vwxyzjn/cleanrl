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
    RepresentationNetwork, Actor, Critic, DynamicsProjector, Dynamics, RewardValueModel, PolicyModel = Model
    _ = envs.reset()
    network = RepresentationNetwork(action_dim=envs.single_action_space.n)
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()
    dynamics_proj = DynamicsProjector()
    dynamics = Dynamics(action_dim=envs.single_action_space.n)
    reward_value_model = RewardValueModel()
    policy_model = PolicyModel(action_dim=envs.single_action_space.n)

    key = jax.random.PRNGKey(seed)
    key, network_key, actor_key, critic_key, dynamics_proj_key, dynamics_key, rv_model_key, p_model_key = jax.random.split(
        key, 8
    )

    example_obs = np.array([envs.single_observation_space.sample()])
    example_carry = RepresentationNetwork.initialize_carry((1,))
    example_reward = np.array([0.0])
    example_action = np.array([envs.single_action_space.sample()])
    network_params = network.init(network_key, example_obs)

    _, example_latent_obs = network.apply(network_params, example_carry, example_obs, example_reward, example_action)
    actor_params = actor.init(actor_key, example_latent_obs)
    critic_params = critic.init(critic_key, example_latent_obs)
    dynamics_proj_params = dynamics_proj.init(critic_key, example_latent_obs)

    example_dyn_carry = dynamics_proj.apply(dynamics_proj_params, example_latent_obs)
    example_dyn_carry = (example_dyn_carry, example_dyn_carry)
    dynamics_params = dynamics.init(dynamics_key, example_dyn_carry, example_action)

    _, example_predicted_latent_obs = dynamics.apply(dynamics_params, example_dyn_carry, example_action)
    reward_value_model_params = reward_value_model.init(rv_model_key, example_predicted_latent_obs)
    policy_model_params = policy_model.init(p_model_key, example_predicted_latent_obs)

    # note: critic_params is not used in this script
    with open(model_path, "rb") as f:
        (
            args,
            (
                network_params,
                actor_params,
                critic_params,
                dynamics_proj_params,
                dynamics_params,
                reward_value_model_params,
                policy_model_params,
            ),
        ) = flax.serialization.from_bytes(
            (
                None,
                (
                    network_params,
                    actor_params,
                    critic_params,
                    dynamics_proj_params,
                    dynamics_params,
                    reward_value_model_params,
                    policy_model_params,
                ),
            ),
            f.read(),
        )

    @jax.jit
    def normalize_logits(logits: jnp.ndarray) -> jnp.ndarray:
        """Normalize thhe logits.

        # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        """
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        logits = logits.clip(min=jnp.finfo(logits.dtype).min)
        return logits

    @jax.jit
    def get_action_and_lstm_carry(
        network_params: flax.core.FrozenDict,
        actor_params: flax.core.FrozenDict,
        obs: np.ndarray,
        lstm_carry: np.ndarray,
        prev_reward: np.ndarray,
        prev_action: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        lstm_carry, hidden = network.apply(network_params, lstm_carry, obs, prev_reward, prev_action)
        logits = normalize_logits(actor.apply(actor_params, hidden))
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        return action, logprob, lstm_carry, key

    # a simple non-vectorized version

    episodic_returns = []
    lstm_hidden = example_carry
    reward = jnp.asarray([0.0])
    action = jnp.asarray([0], jnp.int32)
    for episode in range(eval_episodes):
        episodic_return = 0
        next_obs = envs.reset()
        terminated = False

        if capture_video:
            recorded_frames = []
            # conversion from grayscale into rgb
            recorded_frames.append(cv2.cvtColor(next_obs[0][-1], cv2.COLOR_GRAY2RGB))
        while not terminated:
            action, logprob, lstm_hidden, key = get_action_and_lstm_carry(
                network_params, actor_params, next_obs, lstm_hidden, reward, action, key
            )
            next_obs, reward, _, infos = envs.step(np.array(action))
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

    from cleanrl.muesli_atari_envpool_async_jax_scan_impalanet_machado import (
        Actor,
        Critic,
        Dynamics,
        DynamicsProjector,
        PolicyModel,
        RepresentationNetwork,
        RewardValueModel,
        make_env,
    )

    model_path = hf_hub_download(
        repo_id="vwxyzjn/Pong-v5-muesli_atari_envpool_xla_jax_scan_machado-seed1",
        filename="muesli_atari_envpool_xla_jax_scan_machado.cleanrl_model",
    )
    evaluate(
        model_path,
        make_env,
        "Pong-v5",
        eval_episodes=10,
        run_name=f"eval",
        Model=(RepresentationNetwork, Actor, Critic, DynamicsProjector, Dynamics, RewardValueModel, PolicyModel),
        capture_video=False,
    )
