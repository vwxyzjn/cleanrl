# PPO + Curiosity-Critic, drop-in replacement for the intrinsic reward of
# `ppo_rnd_envpool.py`. PPO/GAE/clip/entropy/value/coef hyperparameters and
# core logging scalars match `ppo_rnd_envpool.py`. The RND target and predictor
# are replaced by a forward-dynamics World Model (WM) and Neural Critic (NC).
# The per-transition Curiosity-Critic objective is:
#
#   e_before = 0.5 * ||theta_t(s_t,a_t) - s_{t+1}||^2
#   theta_{t+1} <- GradStep(theta_t; s_t, a_t, s_{t+1})
#   e_after  = 0.5 * ||theta_{t+1}(s_t,a_t) - s_{t+1}||^2
#   phi_{t+1} <- GradStep(phi_t; s_t, a_t, e_after)
#   r_t = max(0, e_before - phi_{t+1}(s_t, a_t))
#
# PPO, reward normalization, GAE, and auxiliary-model update cadence mirror
# `ppo_rnd_envpool.py`: a rollout's rewards are computed with frozen WM/NC
# snapshots; the WM and NC are then updated on PPO minibatches, so the next
# rollout uses phi after fitting post-WM-update error targets.
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
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

    # Algorithm specific arguments
    env_id: str = "MontezumaRevenge-v5"
    """the id of the environment"""
    total_timesteps: int = 2000000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Curiosity-Critic arguments (kept identical to ppo_rnd_envpool.py's RND args
    # so PPO+RND and PPO+CC share every hyperparameter that controls training.)
    update_proportion: float = 0.25
    """proportion of exp used for World-Model and Neural-Critic update"""
    int_coef: float = 1.0
    """coefficient of extrinsic reward"""
    ext_coef: float = 2.0
    """coefficient of intrinsic reward"""
    int_gamma: float = 0.99
    """Intrinsic reward discount rate"""
    num_iterations_obs_norm_init: int = 50
    """number of iterations to initialize the observations normalization parameters"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic_ext(features + hidden),
            self.critic_int(features + hidden),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


def _action_planes(action_onehot, height, width):
    return action_onehot[:, :, None, None].expand(-1, -1, height, width)


class ForwardCNN(nn.Module):
    """
    Fully convolutional action-conditioned World Model (WM).

    The one-hot action is broadcast as extra spatial channels and concatenated
    with the normalized frame stack. A conv encoder and transposed-conv decoder
    predict the normalized next 84x84 frame.
    """

    # Default channels give 2,203,337 trainable params for 18-action Atari,
    # matching RND's 2,203,296-param predictor within 0.002%.
    def __init__(self, num_actions, frame_stack=4, channels=(184, 240, 120)):
        super().__init__()
        c1, c2, c3 = channels
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels=frame_stack + num_actions, out_channels=c1, kernel_size=8, stride=4)
            ),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            layer_init(
                nn.ConvTranspose2d(in_channels=c3, out_channels=c2, kernel_size=3, stride=1)
            ),
            nn.LeakyReLU(),
            layer_init(
                nn.ConvTranspose2d(in_channels=c2, out_channels=c1, kernel_size=4, stride=2)
            ),
            nn.LeakyReLU(),
            layer_init(
                nn.ConvTranspose2d(in_channels=c1, out_channels=1, kernel_size=8, stride=4)
            ),
        )

    def forward(self, obs_stack, action_onehot):
        _, _, height, width = obs_stack.shape
        action_map = _action_planes(action_onehot, height, width)
        h = torch.cat([obs_stack, action_map], dim=1)
        return self.decoder(self.encoder(h))


class CuriosityCriticCNN(nn.Module):
    """
    Action-conditioned Neural Critic (NC) for the scalar error baseline.

    The NC is trained on the same masked PPO minibatches as the WM, after the WM
    step, to predict the post-update RND-style WM error for each transition.
    """

    # Default channels give 1,677,969 trainable params for 18-action Atari,
    # matching RND's 1,677,984-param target within 0.001%.
    def __init__(self, num_actions, frame_stack=4, channels=(184, 232, 344)):
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels=frame_stack + num_actions, out_channels=c1, kernel_size=8, stride=4)
            ),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=1)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(in_channels=c3, out_channels=1, kernel_size=7, stride=1)),
            nn.Flatten(),
        )

    def forward(self, obs_stack, action_onehot):
        _, _, height, width = obs_stack.shape
        action_map = _action_planes(action_onehot, height, width)
        h = torch.cat([obs_stack, action_map], dim=1)
        return self.encoder(h)


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def _normalize_stack(stack, mean_t, std_t):
    """Normalize with RND's single-channel obs RMS broadcast across frame stacks."""
    return ((stack - mean_t) / std_t).clip(-5, 5).float()


def _rnd_reward_error_per_sample(pred, target):
    """Raw curiosity error, matching RND's 0.5 * squared L2 per sample."""
    return 0.5 * (pred - target).flatten(1).pow(2).sum(1)


def _rnd_update_loss_per_sample(pred, target):
    """Auxiliary-model update metric, matching RND's per-sample MSE."""
    return F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    num_actions = envs.single_action_space.n

    agent = Agent(envs).to(device)
    world_model = ForwardCNN(num_actions=num_actions, frame_stack=4).to(device)
    neural_critic = CuriosityCriticCNN(num_actions=num_actions, frame_stack=4).to(device)

    combined_parameters = list(agent.parameters()) + list(world_model.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )
    # The critic target (post-update WM error) is only available after the WM
    # step completes, so the critic gets its own optimizer.
    critic_optimizer = optim.Adam(
        neural_critic.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_frames = torch.zeros((args.num_steps, args.num_envs, 84, 84)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, _ = envs.step(acs)
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            critic_optimizer.param_groups[0]["lr"] = lrnow

        # obs_rms is constant during the rollout (only updated post-rollout); cache normalization tensors once.
        rollout_obs_mean = torch.from_numpy(obs_rms.mean).to(device)
        rollout_obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_frames[step] = next_obs[:, 3, :, :]

            obs_normalized = _normalize_stack(obs[step], rollout_obs_mean, rollout_obs_std)
            target_next = _normalize_stack(
                next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84),
                rollout_obs_mean,
                rollout_obs_std,
            )
            action_onehot = F.one_hot(action.long(), num_classes=num_actions).float()

            # Match RND rollout timing: compute intrinsic rewards with frozen auxiliary models.
            with torch.no_grad():
                wm_pred = world_model(obs_normalized, action_onehot)
                error_before = _rnd_reward_error_per_sample(wm_pred, target_next)
                critic_pred = neural_critic(obs_normalized, action_onehot).squeeze(-1).clamp(min=0)
                curiosity_rewards[step] = (error_before - critic_pred).clamp(min=0).detach()

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={info['r'][idx]}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_next_frames = next_frames.reshape(-1, 84, 84)

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        update_obs_mean = torch.from_numpy(obs_rms.mean).to(device)
        update_obs_std = torch.sqrt(torch.from_numpy(obs_rms.var).to(device))

        clipfracs = []
        curiosity_critic_loss_sum = 0.0
        curiosity_error_before_sum = 0.0
        curiosity_error_after_sum = 0.0
        curiosity_critic_pred_sum = 0.0
        n_minibatch_steps = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs_normalized = _normalize_stack(b_obs[mb_inds], update_obs_mean, update_obs_std)
                mb_target_next = _normalize_stack(
                    b_next_frames[mb_inds].reshape(-1, 1, 84, 84),
                    update_obs_mean,
                    update_obs_std,
                )
                mb_actions_long = b_actions.long()[mb_inds]
                mb_actions_onehot = F.one_hot(mb_actions_long, num_classes=num_actions).float()

                # WM update mirrors RND predictor training: same minibatches, mask, and MSE metric.
                wm_pred = world_model(mb_obs_normalized, mb_actions_onehot)
                forward_loss = _rnd_update_loss_per_sample(wm_pred, mb_target_next.detach())
                error_before_mb = _rnd_reward_error_per_sample(wm_pred.detach(), mb_target_next).detach()

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], mb_actions_long
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

                with torch.no_grad():
                    post_wm_pred = world_model(mb_obs_normalized, mb_actions_onehot)
                    error_after_mb = _rnd_reward_error_per_sample(post_wm_pred, mb_target_next).detach()

                # NC learns the post-WM-update error baseline on the same masked samples.
                critic_pred_train = neural_critic(mb_obs_normalized, mb_actions_onehot).squeeze(-1)
                critic_loss_per = F.mse_loss(critic_pred_train, error_after_mb, reduction="none")
                critic_loss = (critic_loss_per * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )

                critic_optimizer.zero_grad()
                critic_loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        neural_critic.parameters(),
                        args.max_grad_norm,
                    )
                critic_optimizer.step()

                curiosity_critic_loss_sum += critic_loss.item()
                curiosity_error_before_sum += error_before_mb.mean().item()
                curiosity_error_after_sum += error_after_mb.mean().item()
                curiosity_critic_pred_sum += critic_pred_train.detach().clamp(min=0).mean().item()
                n_minibatch_steps += 1

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # Curiosity-Critic-specific scalars (averaged across all minibatches in this PPO cycle).
        writer.add_scalar("losses/critic_loss", curiosity_critic_loss_sum / max(n_minibatch_steps, 1), global_step)
        writer.add_scalar("losses/error_before", curiosity_error_before_sum / max(n_minibatch_steps, 1), global_step)
        writer.add_scalar("losses/error_after", curiosity_error_after_sum / max(n_minibatch_steps, 1), global_step)
        writer.add_scalar("charts/critic_pred_mean", curiosity_critic_pred_sum / max(n_minibatch_steps, 1), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
