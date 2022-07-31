# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo_dna/#ppo_dna_atari_envpoolpy
import argparse
import os
import random
import time
from collections import deque
from copy import deepcopy
from distutils.util import strtobool

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence
from torch.utils.tensorboard import SummaryWriter


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

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=128,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")

    # DNA policy network optimization hyperparams
    parser.add_argument("--policy-learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the policy optimizer")
    parser.add_argument("--policy-batch-size", type=int, default=2048,
        help="the batch size of the policy optimizer")
    parser.add_argument("--policy-gae-lambda", type=float, default=0.8,
        help="the lambda for the general advantage estimation of policy (zero to disable GAE)")
    parser.add_argument("--policy-update-epochs", type=int, default=2,
        help="the K epochs to update the policy")

    # DNA value network optimization hyperparams
    parser.add_argument("--value-learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the value function optimizer")
    parser.add_argument("--value-batch-size", type=int, default=512,
        help="the batch size of the value function optimizer")
    parser.add_argument("--value-gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation of value function (zero to disable GAE)")
    parser.add_argument("--value-update-epochs", type=int, default=1,
        help="the K epochs to update the value function")

    # DNA value network to policy network distillation hyperparams
    parser.add_argument("--distill-learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the distillation optimizer")
    parser.add_argument("--distill-batch-size", type=int, default=512,
        help="the batch size of the distillation optimizer")
    parser.add_argument("--distill-update-epochs", type=int, default=2,
        help="the K epochs to update the distillation")
    parser.add_argument("--distill-beta", type=float, default=1.0,
        help="distillation policy KL divergence regularization strength")

    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.epoch_size = int(args.num_envs * args.num_steps)
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        # get if the env has lives
        self.has_lives = False
        env.reset()
        info = env.step(np.zeros(self.num_envs, dtype=int))[-1]
        if info["lives"].sum() > 0:
            self.has_lives = True
            print("env has lives")

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
        all_lives_exhausted = infos["lives"] == 0
        if self.has_lives:
            self.episode_returns *= 1 - all_lives_exhausted
            self.episode_lengths *= 1 - all_lives_exhausted
        else:
            self.episode_returns *= 1 - dones
            self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def compute_advantages(rewards, dones, values, next_done, next_value, gamma, gae_lambda):
    total_steps = len(rewards)
    if gae_lambda > 0:
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(total_steps)):
            if t == total_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards)
        for t in reversed(range(total_steps)):
            if t == total_steps - 1:
                nextnonterminal = 1.0 - next_done
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return
        advantages = returns - values
    return advantages, returns


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
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x)).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        value = self.critic(hidden).squeeze(-1)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), probs, value


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
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        seed=args.seed,
    )
    envs.is_vector_env = True
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -3, 3))
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -5, 5))
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent_policy = Agent(envs).to(device)
    agent_value = Agent(envs).to(device)
    policy_optimizer = optim.Adam(agent_policy.parameters(), lr=args.policy_learning_rate, eps=1e-5)
    value_optimizer = optim.Adam(agent_value.parameters(), lr=args.value_learning_rate, eps=1e-5)
    distill_optimizer = optim.Adam(agent_policy.parameters(), lr=args.distill_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.epoch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            policy_optimizer.param_groups[0]["lr"] = frac * args.policy_learning_rate
            value_optimizer.param_groups[0]["lr"] = frac * args.value_learning_rate
            distill_optimizer.param_groups[0]["lr"] = frac * args.distill_learning_rate

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, _, _ = agent_policy.get_action_and_value(next_obs)
                value = agent_value.get_value(next_obs)
                values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent_value.get_value(next_obs).reshape(1, -1)
            advantages, _ = compute_advantages(
                rewards, dones, values, next_done, next_value, args.gamma, args.policy_gae_lambda
            )
            _, returns = compute_advantages(rewards, dones, values, next_done, next_value, args.gamma, args.value_gae_lambda)

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Policy network optimization
        b_inds = np.arange(args.epoch_size)
        clipfracs = []
        for epoch in range(args.policy_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.epoch_size, args.policy_batch_size):
                end = start + args.policy_batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, _, _ = agent_policy.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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

                entropy_loss = entropy.mean()
                policy_loss = pg_loss - args.ent_coef * entropy_loss

                policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(agent_policy.parameters(), args.max_grad_norm)
                policy_optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Value network optimization
        for epoch in range(args.value_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.epoch_size, args.value_batch_size):
                end = start + args.value_batch_size
                mb_inds = b_inds[start:end]

                newvalue = agent_value.get_value(b_obs[mb_inds])

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent_value.parameters(), args.max_grad_norm)
                value_optimizer.step()

        # Value network to policy network distillation
        agent_policy.zero_grad(True)  # don't clone gradients
        old_agent_policy = deepcopy(agent_policy)
        old_agent_policy.eval()
        for epoch in range(args.distill_update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.epoch_size, args.distill_batch_size):
                end = start + args.distill_batch_size
                mb_inds = b_inds[start:end]

                # Compute policy and value targets
                with torch.no_grad():
                    _, _, _, old_action_dist, _ = old_agent_policy.get_action_and_value(b_obs[mb_inds])
                    value_target = agent_value.get_value(b_obs[mb_inds])

                _, _, _, new_action_dist, new_value = agent_policy.get_action_and_value(b_obs[mb_inds])

                # Distillation loss
                policy_kl_loss = kl_divergence(old_action_dist, new_action_dist).mean()
                value_loss = 0.5 * (new_value - value_target).square().mean()
                distill_loss = value_loss + args.distill_beta * policy_kl_loss

                distill_optimizer.zero_grad()
                distill_loss.backward()
                nn.utils.clip_grad_norm_(agent_policy.parameters(), args.max_grad_norm)
                distill_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/policy_learning_rate", policy_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/value_learning_rate", value_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/distill_learning_rate", distill_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/distill_loss", distill_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
