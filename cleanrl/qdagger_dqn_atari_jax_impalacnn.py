# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/qdagger/#qdagger_dqn_atari_jax_impalacnnpy
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from huggingface_hub import hf_hub_download
from rich.progress import track
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from cleanrl.dqn_atari_jax import QNetwork as TeacherModel
from cleanrl_utils.evals.dqn_jax_eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    # QDagger specific arguments
    teacher_policy_hf_repo: str = None
    """the huggingface repo of the teacher policy"""
    teacher_model_exp_name: str = "dqn_atari_jax"
    """the experiment name of the teacher model"""
    teacher_eval_episodes: int = 10
    """the number of episodes to run the teacher policy evaluate"""
    teacher_steps: int = 500000
    """the number of steps to run the teacher policy to generate the replay buffer"""
    offline_steps: int = 500000
    """the number of steps to run the student policy with the teacher's replay buffer"""
    temperature: float = 1.0
    """the temperature parameter for qdagger"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)

        return env

    return thunk


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        inputs = x
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        return x + inputs


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.channels,
            kernel_size=(3, 3),
        )(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        x = ResidualBlock(self.channels)(x)
        return x


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int
    channelss: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        for channels in self.channelss:
            x = ConvSequence(channels)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(channelss=(16, 32, 32), action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, envs.observation_space.sample()),
        target_params=q_network.init(q_key, envs.observation_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    q_network.apply = jax.jit(q_network.apply)

    # QDAGGER LOGIC:
    teacher_model_path = hf_hub_download(
        repo_id=args.teacher_policy_hf_repo, filename=f"{args.teacher_model_exp_name}.cleanrl_model"
    )
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    teacher_model_key = jax.random.PRNGKey(args.seed)
    teacher_params = teacher_model.init(teacher_model_key, envs.observation_space.sample())
    with open(teacher_model_path, "rb") as f:
        teacher_params = flax.serialization.from_bytes(teacher_params, f.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    # evaluate the teacher model
    teacher_episodic_returns = evaluate(
        teacher_model_path,
        make_env,
        args.env_id,
        eval_episodes=args.teacher_eval_episodes,
        run_name=f"{run_name}-teacher-eval",
        Model=TeacherModel,
        epsilon=0.05,
        capture_video=False,
    )
    writer.add_scalar("charts/teacher/avg_episodic_return", np.mean(teacher_episodic_returns), 0)

    # collect teacher data for args.teacher_steps
    # we assume we don't have access to the teacher's replay buffer
    # see Fig. A.19 in Agarwal et al. 2022 for more detail
    teacher_rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        epsilon = linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model.apply(teacher_params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        teacher_rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
        obs = next_obs

    def kl_divergence_with_logits(target_logits, prediction_logits):
        """Implementation of on-policy distillation loss."""
        out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits) - nn.log_softmax(target_logits))
        return jnp.sum(out)

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones, distill_coeff):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        td_target = rewards + (1 - dones) * args.gamma * q_next_target
        teacher_q_values = teacher_model.apply(teacher_params, observations)

        def loss(params, td_target, teacher_q_values, distill_coeff):
            student_q_values = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = student_q_values[np.arange(student_q_values.shape[0]), actions.squeeze()]  # (batch_size,)
            q_loss = ((q_pred - td_target) ** 2).mean()
            teacher_q_values = teacher_q_values / args.temperature
            student_q_values = student_q_values / args.temperature
            distill_loss = jnp.mean(jax.vmap(kl_divergence_with_logits)(teacher_q_values, student_q_values))
            overall_loss = q_loss + distill_coeff * distill_loss
            return overall_loss, (q_loss, q_pred, distill_loss)

        (loss_value, (q_loss, q_pred, distill_loss)), grads = jax.value_and_grad(loss, has_aux=True)(
            q_state.params, td_target, teacher_q_values, distill_coeff
        )
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_loss, q_pred, distill_loss, q_state

    # offline training phase: train the student model using the qdagger loss
    for global_step in track(range(args.offline_steps), description="offline student training"):
        data = teacher_rb.sample(args.batch_size)
        # perform a gradient-descent step
        loss, q_loss, old_val, distill_loss, q_state = update(
            q_state,
            data.observations.numpy(),
            data.actions.numpy(),
            data.next_observations.numpy(),
            data.rewards.flatten().numpy(),
            data.dones.flatten().numpy(),
            1.0,
        )

        # update the target network
        if global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau))

        if global_step % 100 == 0:
            writer.add_scalar("charts/offline/loss", jax.device_get(loss), global_step)
            writer.add_scalar("charts/offline/q_loss", jax.device_get(q_loss), global_step)
            writer.add_scalar("charts/offline/distill_loss", jax.device_get(distill_loss), global_step)

        if global_step % 100000 == 0:
            # evaluate the student model
            model_path = f"runs/{run_name}/{args.exp_name}-offline-{global_step}.cleanrl_model"
            with open(model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(q_state.params))
            print(f"model saved to {model_path}")

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                epsilon=0.05,
            )
            print(episodic_returns)
            writer.add_scalar("charts/offline/avg_episodic_return", np.mean(episodic_returns), global_step)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    obs, _ = envs.reset(seed=args.seed)
    episodic_returns = deque(maxlen=10)
    # online training phase
    for global_step in track(range(args.total_timesteps), description="online student training"):
        global_step += args.offline_steps
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                episodic_returns.append(info["episode"]["r"])
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                if len(episodic_returns) < 10:
                    distill_coeff = 1.0
                else:
                    distill_coeff = max(1 - np.mean(episodic_returns) / np.mean(teacher_episodic_returns), 0)
                loss, q_loss, old_val, distill_loss, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                    distill_coeff,
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/td_loss", jax.device_get(q_loss), global_step)
                    writer.add_scalar("losses/distill_loss", jax.device_get(distill_loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    writer.add_scalar("charts/distill_coeff", distill_coeff, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(distill_coeff)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "Qdagger", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
