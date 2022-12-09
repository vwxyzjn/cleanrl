# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import pybullet_envs  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import glob
import natsort


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
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--run-name", type=str, default=None,
        help="the name of the run to load from")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HopperBulletEnv-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.1,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--num-skills", type=int, default=50)
    parser.add_argument("--discriminator_lr", type=float, default=1e-3,
        help="the learning rate of the discriminator network")
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
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, num_skills):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape) + num_skills, 256
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# INFO: some implementations use separate network for V and Q
# as cleanrl doesn't use a separate V network, we'll follow same structure for DIAYN
# class ValueNetwork(nn.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# INFO: don't need to use OptionsPolicy as it is not used in the paper.
# Instead skill is uniformly sampled from the skills space.
# This can be used later to use pretrained skills to optimize for a specific reward function.
# class OptionsPolicy(nn.Module):
#     def __init__(self, env, num_skills):
#         super().__init__()
#         self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, num_skills)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return OneHotCategorical(logits = x)


class Discriminator(nn.Module):
    def __init__(self, env, num_skills):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_skills)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5
CHECKPOINT_FREQUENCY = 10000


class Actor(nn.Module):
    def __init__(self, env, num_skills):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + num_skills, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def aug_obs_z(obs, skill_one_hot):
    obs = np.asarray(obs)
    aug_obs = np.hstack((obs, skill_one_hot))
    return aug_obs


def split_aug_obs(aug_obs, num_skills):
    # assert type(aug_obs) in [torch.Tensor, np.ndarray] and type(num_skills) is int, "invalid input type"
    obs, skill_one_hot = aug_obs[:, :-num_skills], aug_obs[:, -num_skills:]
    return obs, skill_one_hot


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
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs, args.num_skills).to(device)
    qf1 = SoftQNetwork(envs, args.num_skills).to(device)
    qf2 = SoftQNetwork(envs, args.num_skills).to(device)
    qf1_target = SoftQNetwork(envs, args.num_skills).to(device)
    qf2_target = SoftQNetwork(envs, args.num_skills).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    discriminator = Discriminator(envs, args.num_skills).to(device)
    discriminator_optimizer = optim.Adam(list(discriminator.parameters()), lr=args.discriminator_lr)

    if args.run_name:
        models = natsort.natsorted(glob.glob(f"runs/{args.run_name}/diayn_state_*.pt"))
        model_dict = torch.load(models[-1])
        actor.load_state_dict(model_dict["actor"])
        qf1.load_state_dict(model_dict["qf1"])
        qf2.load_state_dict(model_dict["qf2"])
        qf1_target.load_state_dict(model_dict["qf1_target"])
        qf2_target.load_state_dict(model_dict["qf2_target"])
        discriminator.load_state_dict(model_dict["discriminator"])

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    low = np.hstack([envs.single_observation_space.low, np.full(args.num_skills, 0)])
    high = np.hstack([envs.single_observation_space.high, np.full(args.num_skills, 1)])
    aug_obs_space = gym.spaces.Box(low=low, high=high)
    rb = ReplayBuffer(
        args.buffer_size,
        aug_obs_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()

    # sampling new skill z at the start of an episode
    z = [np.random.randint(args.num_skills) for _ in range(envs.num_envs)]
    one_hot_z = np.zeros((envs.num_envs, args.num_skills), dtype=np.float32)
    for idx in range(envs.num_envs):
        one_hot_z[idx, z[idx]] = 1
    # augmenting observation with skill one hot vector
    z_aug_obs = aug_obs_z(obs, one_hot_z)

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(z_aug_obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        z_aug_obs_next = aug_obs_z(next_obs, one_hot_z)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for idx, info in enumerate(infos):
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return/" + str(z[idx]), info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length/" + str(z[idx]), info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_z_aug_obs_next = z_aug_obs_next.copy()
        for idx, d in enumerate(dones):
            if d:
                real_z_aug_obs_next[idx, : -args.num_skills] = infos[idx]["terminal_observation"]
                # if episode ends, update the sampled skill z for the next episode
                z[idx] = np.random.randint(args.num_skills)
                one_hot_skill = np.zeros(args.num_skills)
                one_hot_skill[z[idx]] = 1
                one_hot_z[idx] = one_hot_skill
                z_aug_obs_next[idx, -args.num_skills :] = one_hot_skill
        rb.add(z_aug_obs, real_z_aug_obs_next, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        z_aug_obs = z_aug_obs_next
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # skill prediction from discriminator
            next_observations, sampled_skills = split_aug_obs(data.next_observations, args.num_skills)
            pred_z = discriminator(next_observations)
            predicted_z_log_probs = F.cross_entropy(pred_z, sampled_skills, reduction="none")

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # use diayn rewards instead of environment rewards
                diayn_reward = - predicted_z_log_probs.detach() - np.log(1 / args.num_skills)
                next_q_value = diayn_reward.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            discriminator_loss = predicted_z_log_probs.mean()
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/discriminator_loss", discriminator_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
            
            if global_step % CHECKPOINT_FREQUENCY == 0:
                model_dict = dict()
                model_dict["actor"] = actor.state_dict()
                model_dict["actor_optimizer"] = actor_optimizer.state_dict()
                model_dict["qf1"] = qf1.state_dict()
                model_dict["qf2"] = qf2.state_dict()
                model_dict["qf1_target"] = qf1_target.state_dict()
                model_dict["qf2_target"] = qf2_target.state_dict()
                model_dict["q_optimizer"] = q_optimizer.state_dict()
                model_dict["discriminator"] = discriminator.state_dict()
                if args.autotune:
                    model_dict["log_alpha"] = log_alpha
                    model_dict["a_optimizer"] = a_optimizer.state_dict()
                model_dict["discriminator_optimizer"] = discriminator_optimizer.state_dict()
                model_dict["global_step"] = global_step
                torch.save(model_dict, f"runs/{run_name}/diayn_state_{global_step}.pt")

    envs.close()
    writer.close()