import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from lil_maze import LilMaze

import wandb


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = -1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "aux hyperparameters optimization" 
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LilMaze"
    """the environment id of the task"""
    total_timesteps: int = 150000
    """total timesteps of the experiments"""
    num_envs: int = 4
    """the number of parallel game environments to run"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""



    # VAE specific arguments
    vae_lr: float = 1e-4
    """the learning rate of the VAE"""
    vae_epochs: int = 1
    """the number of epochs for the VAE"""
    vae_frequency: int = 800
    """the frequency of training VAE"""
    vae_latent_dim: int = 32
    """the latent dimension of the VAE"""
    clip_vae: float = 120.0
    """the clipping of the VAE"""
    vae_batch_size: int = 128
    """the batch size of the VAE"""


    keep_extrinsic_reward: bool = True
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic : float = 100.0
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 1.0
    """the coefficient of the extrinsic reward"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            #env = gym.make(env_id, render_mode="rgb_array")
            env = LilMaze(render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        else:
            env = LilMaze()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
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
    
class VAE(nn.Module):
    def __init__(self, envs, latent_dim, clip_vae=120.0, scale_l = 1000.0):
        super().__init__()
        input_dim = np.prod(envs.single_observation_space.shape)
        self.clip_vae = clip_vae
        self.scale_l = scale_l
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(256, latent_dim)
        self.logstd_layer = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )   
    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logstd = self.logstd_layer(x)
        return mean, logstd

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logstd = self.encode(x/self.scale_l)
        z = mean + torch.randn_like(mean) * torch.exp(logstd)
        x_recon = torch.clamp(self.decode(z), -self.clip_vae, self.clip_vae)
        return x_recon, mean, logstd
    
    def loss(self, x, reduce=True):
        x_recon, mean, logstd = self(x)
        x = x/self.scale_l
        recon_loss = F.mse_loss(x_recon, x, reduction='none').sum(1)
        kl_loss = -0.5 * (1 + 2 * logstd - mean ** 2 - torch.exp(2 * logstd)).sum(1)
        loss = recon_loss + kl_loss
        if reduce:
            return loss.mean()
        return loss

def train(config=None):

    with wandb.init(config=config):

        config = wandb.config

        all_values = {}
        all_values["episodic_return"] = []
        all_values["vae_loss_values"] = []
        all_values["qf1_values"] = []
        all_values["qf2_values"] = []
        all_values["qf1_loss_values"] = []
        all_values["qf2_loss_values"] = []
        all_values["qf_loss_values"] = []
        all_values["actor_loss_values"] = []
        all_values["alpha_values"] = []
        all_values["sps_values"] = []
        all_values["intrinsic_reward_mean_values"] = []
        all_values["intrinsic_reward_max_values"] = []
        all_values["intrinsic_reward_min_values"] = []

                

        for run_index in range(config.number_of_attempts):
            episode_return_values = []
            vae_loss_values = []
            qf1_values = []
            qf2_values = []
            qf1_loss_values = []
            qf2_loss_values = []
            qf_loss_values = []
            actor_loss_values = []
            alpha_values = []
            sps_values = []
            intrinsic_reward_mean_values = []
            intrinsic_reward_max_values = []
            intrinsic_reward_min_values = []

            args = tyro.cli(Args)
            run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

            args.seed = run_index

            # TRY NOT TO MODIFY: seeding
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = args.torch_deterministic

            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

            # env setup
            envs = gym.vector.SyncVectorEnv(
                [make_env(args.env_id, args.seed, i, args.capture_video, run_name) for i in range(args.num_envs)]
            )
            assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

            max_action = float(envs.single_action_space.high[0])

            actor = Actor(envs).to(device)
            qf1 = SoftQNetwork(envs).to(device)
            qf2 = SoftQNetwork(envs).to(device)
            qf1_target = SoftQNetwork(envs).to(device)
            qf2_target = SoftQNetwork(envs).to(device)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())
            q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
            actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

            vae = VAE(envs, 
                    latent_dim=args.vae_latent_dim, 
                    clip_vae=args.clip_vae).to(device)
            vae_optimizer = optim.Adam(vae.parameters(), lr=config.vae_lr, eps=1e-5)

            # Automatic entropy tuning
            if args.autotune:
                target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
                log_alpha = torch.zeros(1, requires_grad=True, device=device)
                alpha = log_alpha.exp().item()
                a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
            else:
                alpha = args.alpha

            envs.single_observation_space.dtype = np.float32

            # The replay buffer parameters have been updated to handle multiple envs
            rb = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                handle_timeout_termination=False,
                n_envs=args.num_envs
            )
            start_time = time.time()

            # TRY NOT TO MODIFY: start the game
            obs, _ = envs.reset(seed=args.seed)
            for global_step in range(args.total_timesteps):
                # ALGO LOGIC: put action logic here
                if global_step < args.learning_starts:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                    actions = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        
                        break

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        real_next_obs[idx] = infos["final_observation"][idx]
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

                # ALGO LOGIC: training.
                if global_step > args.learning_starts:
                    mean_vae_loss = 0.0
                    if global_step % config.vae_frequency == 0:
                        mean_vae_loss = 0.0
                        for _ in range(args.vae_epochs):
                            data = rb.sample(args.batch_size)
                            
                            vae_loss = vae.loss(data.observations, reduce=True)
                            vae_optimizer.zero_grad()
                            vae_loss.backward()
                            vae_optimizer.step()
                            mean_vae_loss += vae_loss.item()

                        mean_vae_loss /= args.vae_epochs
                        


                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        intrinsic_reward = vae.loss(data.observations, reduce = False)
                        extrinsic_reward = data.rewards.flatten()
                        if args.keep_extrinsic_reward:
                            rewards = extrinsic_reward*config.coef_extrinsic + intrinsic_reward * config.coef_intrinsic
                        else:
                            rewards = intrinsic_reward.flatten() *config.coef_intrinsic
                        next_q_value = rewards + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = qf2(data.observations, data.actions).view(-1)


                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the model
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                        for _ in range(
                            args.policy_frequency
                        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                            pi, log_pi, _ = actor.get_action(data.observations)
                            qf1_pi = qf1(data.observations, pi)
                            qf2_pi = qf2(data.observations, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = actor.get_action(data.observations)
                                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

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
                        
                        episode_return_values.append(info["episode"]["r"])
                        vae_loss_values.append(mean_vae_loss)
                        qf1_values.append(qf1_a_values.mean().item())
                        qf2_values.append(qf2_a_values.mean().item())
                        qf1_loss_values.append(qf1_loss.item())
                        qf2_loss_values.append(qf2_loss.item())
                        qf_loss_values.append(qf_loss.item() / 2.0)
                        actor_loss_values.append(actor_loss.item())
                        alpha_values.append(alpha)
                        sps_values.append(int(global_step / (time.time() - start_time)))
                        intrinsic_reward_mean_values.append(intrinsic_reward.mean().item())
                        intrinsic_reward_max_values.append(intrinsic_reward.max().item())
                        intrinsic_reward_min_values.append(intrinsic_reward.min().item())

                                        
                        
            envs.close()
            all_values["episodic_return"].append(episode_return_values)
            all_values["vae_loss_values"].append(vae_loss_values)
            all_values["qf1_values"].append(qf1_values)
            all_values["qf2_values"].append(qf2_values)
            all_values["qf1_loss_values"].append(qf1_loss_values)
            all_values["qf2_loss_values"].append(qf2_loss_values)
            all_values["qf_loss_values"].append(qf_loss_values)
            all_values["actor_loss_values"].append(actor_loss_values)
            all_values["alpha_values"].append(alpha_values)
            all_values["sps_values"].append(sps_values)
            all_values["intrinsic_reward_mean_values"].append(intrinsic_reward_mean_values)
            all_values["intrinsic_reward_max_values"].append(intrinsic_reward_max_values)
            all_values["intrinsic_reward_min_values"].append(intrinsic_reward_min_values)
        
        mean = {}
        std = {}
        for key in all_values:
            all_values[key] = np.array(all_values[key])
            mean[key] = np.mean(all_values[key], axis=0)
            std[key] = np.std(all_values[key], axis=0)

            for step, (mean_, std_) in enumerate(zip(mean[key], std[key])):
                if key == "episodic_return":
                    wandb.log({
                        f"{key}": mean_,
                        f"{key}_upper": mean_ + std_,
                        f"{key}_lower": mean_ - std_
                    }, step=step)
                else:
                    wandb.log({
                        f"{key}": mean_
                    }, step=step)
            
    


wandb.agent("w18lvp6x", train, project="aux hyperparameters optimization", count=10)