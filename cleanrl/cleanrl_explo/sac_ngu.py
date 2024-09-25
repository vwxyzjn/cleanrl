# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 12
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "SAC - exploration with NGU" 
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 200000
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



    # NGU specific arguments
    ngu_lr: float = 0.00004501
    """the learning rate of the NGU"""
    ngu_epochs: int = 4
    """the number of epochs for the NGU"""
    ngu_frequency: int = 900
    """the frequency of training NGU"""
    ngu_feature_dim: int = 64
    """the feature dimension of the NGU"""
    k_nearest: int = 6
    """the number of nearest neighbors for the NGU"""
    clip_reward: float = 0.3656
    """the clipping value of the reward"""
    c: float = 0.001
    """the constant used not to divide by zero"""
    L: float = 5.0
    """the maximum value for the multiplier in the intrinsic reward of NGU"""
    epsilon_kernel: float = 1e-3
    """the epsilon value for the kernel of the NGU"""


    keep_extrinsic_reward: bool = False
    """if toggled, the extrinsic reward will be kept"""
    coef_intrinsic : float = 48.311
    """the coefficient of the intrinsic reward"""
    coef_extrinsic : float = 7.099
    """the coefficient of the extrinsic reward"""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

class NGU_ReplayBuffer():
    def __init__(self, buffer_size, observation_space, action_space, device, handle_timeout_termination=False, n_envs=1):
        self.buffer_size = buffer_size
        self.device = device
        self.handle_timeout_termination = handle_timeout_termination
        self.n_envs = n_envs

        self.observations = np.zeros((buffer_size, n_envs) + observation_space.shape, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, n_envs) + observation_space.shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_envs) + action_space.shape, dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards_ngu = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, buffer_size

    def add(self, obs, next_obs, action, reward, reward_ngu, done, info):
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.rewards_ngu[self.ptr] = reward_ngu
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs_2 = np.random.randint(0, self.n_envs, size=batch_size)
        return (
            torch.as_tensor(self.observations[idxs,idxs_2,:], device=self.device),
            torch.as_tensor(self.next_observations[idxs,idxs_2,:], device=self.device),
            torch.as_tensor(self.actions[idxs,idxs_2,:], device=self.device),
            torch.as_tensor(self.rewards[idxs,idxs_2], device=self.device),
            torch.as_tensor(self.rewards_ngu[idxs,idxs_2], device=self.device),
            torch.as_tensor(self.dones[idxs,idxs_2], device=self.device),
        )

    def __len__(self):
        return self.size


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
    
class NGU(nn.Module):
    def __init__(self, envs, feature_dim, k_nearest, clip_reward, c, L, epsilon_kernel):
        super().__init__()
        state_dim = np.prod(envs.single_observation_space.shape)
        action_dim = np.prod(envs.single_action_space.shape)
        self.feature_dim = feature_dim
        self.k_nearest = k_nearest
        self.clip_reward = clip_reward
        self.c = c
        self.L = L
        self.epsilon_kernel = epsilon_kernel

        # RND
        # trained network
        self.f1 = nn.Linear(state_dim, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, 1)
        # target network
        self.f1_t = nn.Linear(state_dim, 128)
        self.f2_t = nn.Linear(128, 64)
        self.f3_t = nn.Linear(64, 1)
        # embedding network
        self.f1_z = nn.Linear(state_dim, 128)
        self.f2_z = nn.Linear(128, 64)
        self.f3_z = nn.Linear(64, feature_dim)
        # action network
        self.f1_a = nn.Linear(feature_dim*2 , 128)
        self.f2_a = nn.Linear(128, 64)
        self.f3_a = nn.Linear(64, action_dim)
        # running average of the squared Euclidean distance of the k-th nearest neighbors
        self.dm2 = 0.0

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x

    def forward_t(self, x):
        with torch.no_grad():
            x = F.relu(self.f1_t(x))
            x = F.relu(self.f2_t(x))
            x = self.f3_t(x)
            return x
    
    def rnd_loss(self, x, reduce = True):
        return F.mse_loss(self.forward(x), self.forward_t(x)) if reduce else F.mse_loss(self.forward(x), self.forward_t(x), reduction = 'none')
    
    def embedding(self, s):
        x = F.relu(self.f1_z(s))
        x = F.relu(self.f2_z(x))
        x = self.f3_z(x)
        return x
    
    def action_pred(self, s0, s1):
        x = torch.cat([s0, s1], 1)
        x = F.relu(self.f1_a(x))
        x = F.relu(self.f2_a(x))
        x = self.f3_a(x)
        return x
    
    def reward_episode(self, s, episode):
        z_s = self.embedding(s)
        z_episode = self.embedding(episode)

        dist = torch.norm(z_s - z_episode, dim=1)
        kernel = self.epsilon_kernel/(dist/self.dm2 + self.epsilon_kernel)
        top_k_kernel = torch.topk(kernel, self.k_nearest, largest = True)
        top_k = torch.topk(dist, self.k_nearest, largest = False)
        self.dm2 = 0.99 * self.dm2 + 0.01 * top_k.values.mean().item()
        reward_episodic = (1/(torch.sqrt(top_k_kernel.values.mean()) + self.c)).item() 

        return reward_episodic
    
    

    def loss(self,s,s_next,a,d): 
        rnd_loss = self.rnd_loss(s)

        s0 = self.embedding(s)
        s1 = self.embedding(s_next)
        h_loss = torch.norm(self.action_pred(s0, s1) - a, dim=1) * (1-d)

        return rnd_loss + h_loss.mean()

def main(seed=None, sweep=False):

    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    if seed is not None:
        args.seed = seed
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"


    # For hyperparameter optimization, see trainer.py file
    if sweep:
        episodic_returns_list = []
        corresponding_steps = []

        import wandb
        wandb.init()

        config = wandb.config

        for key, value in vars(args).items():
            if key in config:
                setattr(args, key, config[key])


    else :
        
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

    ngu = NGU(envs, 
              feature_dim = args.ngu_feature_dim,
              k_nearest = args.k_nearest,
              clip_reward = args.clip_reward,
              c = args.c,
              L = args.L,
              epsilon_kernel = args.epsilon_kernel
              ).to(device)
    ngu_optimizer = optim.Adam(ngu.parameters(), lr=args.ngu_lr)
    episodes = [ [] for _ in range(args.num_envs)]

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32


    # This replay buffer is hand designed for NGU
    # The replay buffer parameters have been updated to handle multiple envs
    rb = NGU_ReplayBuffer(
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

        # COMPUTE REWARD
        reward_ngu = torch.zeros(args.num_envs)
        for idx in range(args.num_envs):
            with torch.no_grad():
                reward_ngu[idx] = ngu.reward_episode(torch.tensor(obs[idx]).unsqueeze(0).float().to(device), torch.tensor(np.array(episodes[idx])).float().to(device)) if len(episodes[idx]) > args.k_nearest else 0.0
        

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    if sweep:
                        episodic_returns_list.append(info["episode"]["r"])
                        corresponding_steps.append(global_step)
                    else:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, (done_, trunc) in enumerate(zip(terminations,truncations)):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
            if done_ or trunc:
                episodes[idx] = []
        rb.add(obs, real_next_obs, actions, rewards, reward_ngu, terminations, infos)

        for idx, ob in enumerate(obs):
            episodes[idx].append(ob)
            

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            if global_step % args.ngu_frequency == 0:
                mean_ngu_loss = 0.0
                for _ in range(args.ngu_epochs):
                    data = rb.sample(args.batch_size)
                    data_observations = data[0]
                    data_next_observations = data[1]
                    data_actions = data[2]
                    data_rewards = data[3]
                    data_rewards_ngu = data[4]
                    data_dones = data[5]
                    
                    ngu_loss = ngu.loss(data_observations, data_next_observations, data_actions, data_dones)
                    ngu_optimizer.zero_grad()
                    ngu_loss.backward()
                    ngu_optimizer.step()
                    mean_ngu_loss += ngu_loss.item()

                mean_ngu_loss /= args.ngu_epochs
                if not sweep:
                    writer.add_scalar("losses/ngu_loss", mean_ngu_loss, global_step)
                


            data = rb.sample(args.batch_size)
            data_observations = data[0]
            data_next_observations = data[1]
            data_actions = data[2]
            data_rewards = data[3]
            data_rewards_ngu = data[4]
            data_dones = data[5]
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data_next_observations)
                qf1_next_target = qf1_target(data_next_observations, next_state_actions)
                qf2_next_target = qf2_target(data_next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                rnd_loss = ngu.rnd_loss(data_observations, reduce = False)
                intrinsic_reward = data_rewards_ngu * torch.min(torch.max(rnd_loss.flatten(), torch.tensor(1).to(device)), torch.tensor(args.L).to(device))
                intrinsic_reward = torch.clip(intrinsic_reward, -args.clip_reward, args.clip_reward)
                extrinsic_reward = data_rewards.flatten()
                if args.keep_extrinsic_reward:
                    rewards = extrinsic_reward*args.coef_extrinsic + intrinsic_reward*args.coef_intrinsic
                else:
                    rewards = intrinsic_reward *args.coef_intrinsic
                next_q_value = rewards + (1 - data_dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                

            qf1_a_values = qf1(data_observations, data_actions).view(-1)
            qf2_a_values = qf2(data_observations, data_actions).view(-1)


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
                    pi, log_pi, _ = actor.get_action(data_observations)
                    qf1_pi = qf1(data_observations, pi)
                    qf2_pi = qf2(data_observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data_observations)
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

            if global_step % 100 == 0 and not sweep:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar("specific/intrinsic_reward_mean", intrinsic_reward.mean().item(), global_step)
                writer.add_scalar("specific/intrinsic_reward_max", intrinsic_reward.max().item(), global_step)
                writer.add_scalar("specific/intrinsic_reward_min", intrinsic_reward.min().item(), global_step)
                
    envs.close()
    if sweep:
        return episodic_returns_list, corresponding_steps
    writer.close()

if __name__ == "__main__":
    main()