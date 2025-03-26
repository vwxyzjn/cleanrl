# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rainbow/#rainbow_ataripy
import os
import random
import time
import math
import collections
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
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
    
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0000625
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 8000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    n_step: int = 3
    """the number of steps to look ahead for n-step Q learning"""
    prioritized_replay_alpha: float = 0.5
    """alpha parameter for prioritized replay buffer"""
    prioritized_replay_beta: float = 0.4
    """beta parameter for prioritized replay buffer"""
    prioritized_replay_eps: float = 1e-6
    """epsilon parameter for prioritized replay buffer"""
    n_atoms: int = 51
    """the number of atoms"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""

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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        # factorized gaussian noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

# ALGO LOGIC: initialize agent here:
class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super(NoisyDuelingDistributionalNetwork, self).__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.single_action_space.n
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 3136

        self.value_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512), 
            nn.ReLU(),
            NoisyLinear(512, n_atoms)
        )
        
        self.advantage_head = nn.Sequential(
            NoisyLinear(conv_output_size, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms * self.n_actions)
        )

    def forward(self, x):
        h = self.network(x / 255.0)
        value = self.value_head(h).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(h).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_dist = F.softmax(q_atoms, dim=2)
        return q_dist

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
              
PrioritizedBatch = collections.namedtuple(
    'PrioritizedBatch',
    ['observations', 'actions', 'rewards', 'next_observations', 'dones', 'indices', 'weights']
)

class SumSegmentTree:
    def __init__(self, capacity):
        self.n = capacity
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float32)

    def update(self, idx, value):
        idx += self.tree_size
        self.tree[idx] = value
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def total(self):
        return self.tree[1]

    def retrieve(self, value):
        idx = 1
        while idx < self.tree_size:
            left = 2 * idx
            if self.tree[left] >= value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        result = idx - self.tree_size
        return min(result, self.n - 1)

class MinSegmentTree:
    def __init__(self, capacity):
        self.n = capacity  # actual capacity
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        self.tree = np.full(2 * self.tree_size, float('inf'), dtype=np.float32)

    def update(self, idx, value):
        idx += self.tree_size
        self.tree[idx] = value
        while idx > 1:
            idx //= 2
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])

    def min(self, start=0, end=None):
        if end is None:
            end = self.n  # only consider the actual capacity
        m = float('inf')
        start += self.tree_size
        end += self.tree_size
        while start < end:
            if start % 2 == 1:
                m = min(m, self.tree[start])
                start += 1
            if end % 2 == 1:
                end -= 1
                m = min(m, self.tree[end])
            start //= 2
            end //= 2
        return m

class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, device, n_step, gamma, alpha, beta, eps):
        self.capacity = capacity
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.obs_shape = obs_shape

        self.buffer_obs = np.empty((capacity, ) + obs_shape, dtype=np.uint8)
        self.buffer_next_obs = np.empty((capacity, ) + obs_shape, dtype=np.uint8)
        self.buffer_actions = np.empty((capacity,), dtype=np.int64)
        self.buffer_rewards = np.empty((capacity,), dtype=np.float32)
        self.buffer_dones = np.empty((capacity,), dtype=np.bool_)

        self.pos = 0
        self.size = 0

        self.n_step_buffer = deque(maxlen=n_step)

        self.max_priority = 1.0

        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)

        total_bytes = (
            self.buffer_obs.nbytes +
            self.buffer_next_obs.nbytes +
            self.buffer_actions.nbytes +
            self.buffer_rewards.nbytes +
            self.buffer_dones.nbytes
        )
        print(f"Replay Buffer will use approximately {total_bytes / (1024*1024):.2f} MB of RAM.")

    def add(self, obs, next_obs, actions, rewards, dones):
        for o, no, a, r, d in zip(obs, next_obs, actions, rewards, dones):
            transition = (o, a, r, no, d)
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_step:
                # if episode ends before filling the n-step buffer, flush what we have
                if d:
                    self._flush_n_step_buffer()
                continue
            o0, a0, _, _, _ = self.n_step_buffer[0]
            cum_reward, final_next, final_done = self._get_n_step_info()
            self._add(o0, a0, cum_reward, final_next, final_done)
            # cut n_step returns accross episodes
            if d:
                self._flush_n_step_buffer()

    def _get_n_step_info(self):
        cum_reward = 0.0
        for idx, (_, _, r, next_obs, done) in enumerate(self.n_step_buffer):
            cum_reward += (self.gamma ** idx) * r
            if done:
                break
        final_next = self.n_step_buffer[idx][3]
        final_done = self.n_step_buffer[idx][4]
        return cum_reward, final_next, final_done

    def _flush_n_step_buffer(self):
        while self.n_step_buffer:
            o, a, r, next_obs, d = self.n_step_buffer.popleft()
            cum_reward = 0.0
            transitions = [(o, a, r, next_obs, d)] + list(self.n_step_buffer)
            final_next, final_done = next_obs, d
            for idx, (_, _, r_i, next_obs_i, done_i) in enumerate(transitions):
                cum_reward += (self.gamma ** idx) * r_i
                final_next, final_done = next_obs_i, done_i
                if done_i:
                    break
            self._add(o, a, cum_reward, final_next, final_done)

    def _add(self, obs, action, reward, next_obs, done):
        """Adds a single (n-step) transition into the buffer and updates the trees."""
        idx = self.pos
        self.buffer_obs[idx] = obs
        self.buffer_next_obs[idx] = next_obs
        self.buffer_actions[idx] = action
        self.buffer_rewards[idx] = reward
        self.buffer_dones[idx] = done

        # Use the maximum priority so far for new transitions.
        priority = self.max_priority ** self.alpha
        self.sum_tree.update(idx, priority)
        self.min_tree.update(idx, priority)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Batch size larger than the buffer size.")
        indices = []
        total_priority = self.sum_tree.total()
        segment = total_priority / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            sample_val = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(sample_val)
            indices.append(idx)
        indices = np.array(indices)

        obs_batch = torch.from_numpy(self.buffer_obs[indices]).to(self.device).float()
        actions_batch = torch.from_numpy(self.buffer_actions[indices]).to(self.device).long().unsqueeze(1)
        rewards_batch = torch.from_numpy(self.buffer_rewards[indices]).to(self.device).float().unsqueeze(1)
        next_obs_batch = torch.from_numpy(self.buffer_next_obs[indices]).to(self.device).float()
        dones_batch = torch.from_numpy(self.buffer_dones[indices]).to(self.device).float().unsqueeze(1)

        probs = np.array([self.sum_tree.tree[idx + self.sum_tree.tree_size] for idx in indices])
        p = probs / total_priority
        weights = (p * self.size) ** (-self.beta)
        min_prob = self.min_tree.min(0, self.size) / total_priority
        max_weight = (min_prob * self.size) ** (-self.beta)
        weights = weights / max_weight
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)

        return PrioritizedBatch(obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch, indices, weights)

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            updated_priority = abs(priority) + self.eps
            self.max_priority = max(self.max_priority, updated_priority)
            prio = updated_priority ** self.alpha
            self.sum_tree.update(idx, prio)
            self.min_tree.update(idx, prio)

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

    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
        args.n_step,
        args.gamma,
        args.prioritized_replay_alpha,
        args.prioritized_replay_beta,
        args.prioritized_replay_eps,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # anneal PER beta to 1
        rb.beta = min(1.0, args.prioritized_replay_beta + global_step * (1.0 - args.prioritized_replay_beta) / args.total_timesteps)
        
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            q_dist = q_network(torch.Tensor(obs).to(device))
            q_values = torch.sum(q_dist * q_network.support, dim=2)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                q_network.reset_noise()
                # should we reset the noise for the target network?
                target_network.reset_noise()
                data = rb.sample(args.batch_size)
                
                with torch.no_grad():
                    next_dist = target_network(data.next_observations)  # [B, num_actions, n_atoms]
                    support = target_network.support  # [n_atoms]
                    next_q_values = torch.sum(next_dist * support, dim=2)  # [B, num_actions]

                    # double q-learning
                    next_dist_online = q_network(data.next_observations)  # [B, num_actions, n_atoms]
                    next_q_online = torch.sum(next_dist_online * support, dim=2)  # [B, num_actions]
                    best_actions = torch.argmax(next_q_online, dim=1)  # [B]
                    next_pmfs = next_dist[torch.arange(args.batch_size), best_actions]  # [B, n_atoms]
                    
                    # compute the n-step Bellman update.
                    gamma_n = args.gamma ** args.n_step
                    next_atoms = data.rewards + gamma_n * support * (1 - data.dones)
                    tz = next_atoms.clamp(q_network.v_min, q_network.v_max)
                    
                    # projection
                    delta_z = q_network.delta_z  
                    b = (tz - q_network.v_min) / delta_z  # shape: [B, n_atoms]
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj= 2, and lj= 1
                    d_m_l = (u.float() + (l == b).float() - b) * next_pmfs  # [B, n_atoms]
                    d_m_u = (b - l) * next_pmfs  # [B, n_atoms]
                    
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])
                        
                dist = q_network(data.observations)  # [B, num_actions, n_atoms]
                pred_dist = dist.gather(1, data.actions.unsqueeze(-1).expand(-1, -1, args.n_atoms)).squeeze(1)
                log_pred = torch.log(pred_dist.clamp(min=1e-5, max=1 - 1e-5))

                loss_per_sample = -(target_pmfs * log_pred).sum(dim=1)
                loss = (loss_per_sample * data.weights.squeeze()).mean()

                # update priorities
                new_priorities = loss_per_sample.detach().cpu().numpy()
                rb.update_priorities(data.indices, new_priorities)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    q_values = (pred_dist * q_network.support).sum(dim=1)  # [B]
                    writer.add_scalar("losses/q_values", q_values.mean().item(), global_step)
                    sps = int(global_step / (time.time() - start_time))
                    print("SPS:", sps)
                    writer.add_scalar("charts/SPS", sps, global_step)
                    writer.add_scalar("charts/beta", rb.beta, global_step)
                    
                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1.0 - args.tau) * target_param.data)

    envs.close()
    writer.close()
