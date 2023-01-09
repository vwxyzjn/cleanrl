# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_atari_jaxpy
import argparse
from collections import deque
import os
import random
import time
from distutils.util import strtobool
from typing import Sequence

os.environ[
    "XLA_PYTHON_CLIENT_MEM_FRACTION"
] = "0.7"  # see https://github.com/google/jax/discussions/6332#discussioncomment-1279991

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
from rich.progress import track
from huggingface_hub import hf_hub_download
from cleanrl.dqn_atari_jax import QNetwork as TeacherModel
from cleanrl_utils.evals.dqn_jax_eval import evaluate

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")

    # QDagger specific arguments
    parser.add_argument("--teacher-policy-hf-repo", type=str, default="cleanrl/BreakoutNoFrameskip-v4-dqn_atari_jax-seed1",
        help="the huggingface repo of the teacher policy")
    parser.add_argument("--teacher-steps", type=int, default=500000,
        help="the number of steps to run the teacher policy to generate the replay buffer")
    parser.add_argument("--offline-steps", type=int, default=500000,
        help="the number of steps to run the student policy with the teacher's replay buffer")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="the temperature parameter for qdagger")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
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
    args = parse_args()
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = envs.reset()

    q_network = QNetwork(channelss=(16, 32, 32), action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)

    # QDAGGER LOGIC:
    teacher_model_path = hf_hub_download(repo_id=args.teacher_policy_hf_repo, filename="dqn_atari_jax.cleanrl_model")
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    teacher_model_key = jax.random.PRNGKey(args.seed)
    teacher_params = teacher_model.init(teacher_model_key, obs)
    with open(teacher_model_path, "rb") as f:
        teacher_params = flax.serialization.from_bytes(teacher_params, f.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    # evaluate the teacher model
    teacher_episodic_returns = evaluate(
        teacher_model_path,
        make_env,
        args.env_id,
        eval_episodes=10,
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
        handle_timeout_termination=True,
    )
    
    obs = envs.reset()
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        epsilon = linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model.apply(teacher_params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)
        next_obs, rewards, dones, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        teacher_rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        obs = next_obs



    def kl_divergence_with_logits(target_logits, prediction_logits):
        """Implementation of on-policy distillation loss."""
        out = -nn.softmax(target_logits) * (nn.log_softmax(prediction_logits)
                                            - nn.log_softmax(target_logits))
        return jnp.sum(out)

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones, distill_coeff):
        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target
        teacher_q_values = teacher_model.apply(teacher_params, observations)

        def loss(params, next_q_value, teacher_q_values, distill_coeff):
            student_q_values = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = student_q_values[np.arange(student_q_values.shape[0]), actions.squeeze()]  # (batch_size,)
            q_loss = ((q_pred - next_q_value) ** 2).mean()
            teacher_q_values = teacher_q_values / args.temperature
            student_q_values = student_q_values / args.temperature
            distill_loss = jnp.mean(jax.vmap(kl_divergence_with_logits)(teacher_q_values, student_q_values))
            overall_loss = q_loss + distill_coeff * distill_loss
            return overall_loss, (q_loss, q_pred, distill_loss)

        (loss_value, (q_loss, q_pred, distill_loss)), grads = jax.value_and_grad(loss, has_aux=True)(q_state.params, next_q_value, teacher_q_values, distill_coeff)
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

    # TODO: question: do we need to start the student rb from scratch?
    rb = teacher_rb
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])
    obs = envs.reset()
    episodic_returns = deque(maxlen=10)
    for global_step in track(range(args.total_timesteps), description="online student training"):
        global_step += args.offline_steps
        # ALGO LOGIC: put action logic here
        # # TODO: question: do we need to use epsilon greedy here?
        # epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                episodic_returns.append(info["episode"]["r"])
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

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
                    distill_coeff
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/td_loss", jax.device_get(q_loss), global_step)
                    writer.add_scalar("losses/distill_loss", jax.device_get(distill_loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    writer.add_scalar("charts/distill_coeff", distill_coeff, global_step)
                    print(distill_coeff)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update the target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    # if args.save_model:
    #     model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
    #     with open(model_path, "wb") as f:
    #         f.write(flax.serialization.to_bytes(q_state.params))
    #     print(f"model saved to {model_path}")
    #     from cleanrl_utils.evals.dqn_jax_eval import evaluate

    #     episodic_returns = evaluate(
    #         model_path,
    #         make_env,
    #         args.env_id,
    #         eval_episodes=10,
    #         run_name=f"{run_name}-eval",
    #         Model=QNetwork,
    #         epsilon=0.05,
    #     )
    #     for idx, episodic_return in enumerate(episodic_returns):
    #         writer.add_scalar("eval/episodic_return", episodic_return, idx)

    #     if args.upload_model:
    #         from cleanrl_utils.huggingface import push_to_hub

    #         repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #         repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #         push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
