# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/qdagger/#qdagger_dqn_atari_jax_impalacnnpy
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

from cleanrl.dqn_atari import QNetwork as TeacherModel
from cleanrl_utils.evals.dqn_eval import evaluate


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
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
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
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
    parser.add_argument("--teacher-policy-hf-repo", type=str, default=None,
        help="the huggingface repo of the teacher policy")
    parser.add_argument("--teacher-eval-episodes", type=int, default=10,
        help="the number of episodes to run the teacher policy evaluate")
    parser.add_argument("--teacher-steps", type=int, default=500000,
        help="the number of steps to run the teacher policy to generate the replay buffer")
    parser.add_argument("--offline-steps", type=int, default=500000,
        help="the number of steps to run the student policy with the teacher's replay buffer")
    parser.add_argument("--temperature", type=float, default=1.0,
        help="the temperature parameter for qdagger")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-dqn_atari-seed1"

    return args


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
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        c, h, w = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=env.single_action_space.n),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def kl_divergence_with_logits(target_logits, prediction_logits):
    """Implementation of on-policy distillation loss."""
    out = -F.softmax(target_logits, dim=-1) * (F.log_softmax(prediction_logits, dim=-1) - F.log_softmax(target_logits, dim=-1))
    return torch.sum(out)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # QDAGGER LOGIC:
    teacher_model_path = hf_hub_download(repo_id=args.teacher_policy_hf_repo, filename="dqn_atari.cleanrl_model")
    teacher_model = TeacherModel(envs).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()

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
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    for global_step in track(range(args.teacher_steps), description="filling teacher's replay buffer"):
        epsilon = linear_schedule(args.start_e, args.end_e, args.teacher_steps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = teacher_model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        teacher_rb.add(obs, real_next_obs, actions, rewards, terminated, infos)
        obs = next_obs

    # offline training phase: train the student model using the qdagger loss
    for global_step in track(range(args.offline_steps), description="offline student training"):
        data = teacher_rb.sample(args.batch_size)
        # perform a gradient-descent step
        with torch.no_grad():
            target_max, _ = target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            teacher_q_values = teacher_model(data.observations) / args.temperature

        student_q_values = q_network(data.observations)
        old_val = student_q_values.gather(1, data.actions).squeeze()
        q_loss = F.mse_loss(td_target, old_val)

        student_q_values = student_q_values / args.temperature
        distill_loss = torch.mean(kl_divergence_with_logits(teacher_q_values, student_q_values))

        loss = q_loss + 1.0 * distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data)

        if global_step % 100 == 0:
            writer.add_scalar("charts/offline/loss", loss, global_step)
            writer.add_scalar("charts/offline/q_loss", q_loss, global_step)
            writer.add_scalar("charts/offline/distill_loss", distill_loss, global_step)

        if global_step % 100000 == 0:
            # evaluate the student model
            model_path = f"runs/{run_name}/{args.exp_name}-offline-{global_step}.cleanrl_model"
            torch.save(q_network.state_dict(), model_path)
            print(f"model saved to {model_path}")

            episodic_returns = evaluate(
                model_path,
                make_env,
                args.env_id,
                eval_episodes=10,
                run_name=f"{run_name}-eval",
                Model=QNetwork,
                device=device,
                epsilon=0.05,
            )
            print(episodic_returns)
            writer.add_scalar("charts/offline/avg_episodic_return", np.mean(episodic_returns), global_step)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
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
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

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
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    teacher_q_values = teacher_model(data.observations) / args.temperature

                student_q_values = q_network(data.observations)
                old_val = student_q_values.gather(1, data.actions).squeeze()
                q_loss = F.mse_loss(td_target, old_val)

                student_q_values = student_q_values / args.temperature
                distill_loss = torch.mean(kl_divergence_with_logits(teacher_q_values, student_q_values))

                loss = q_loss + distill_coeff * distill_loss

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss, global_step)
                    writer.add_scalar("losses/td_loss", q_loss, global_step)
                    writer.add_scalar("losses/distill_loss", distill_loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/distill_coeff", distill_coeff, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    print(distill_coeff)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
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
