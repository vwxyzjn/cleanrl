# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Sequence

import envpool
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=64,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=2,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


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
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
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


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(64)(x)
        critic = nn.tanh(critic)
        critic = nn.Dense(64)(critic)
        critic = nn.tanh(critic)
        critic = nn.Dense(1)(critic)
        return critic


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(64)(x)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(64)(actor_mean)
        actor_mean = nn.tanh(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        actor_logstd = jnp.zeros((1, self.action_dim))
        return actor_mean, actor_logstd


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
    jaxRNG = jax.random.PRNGKey(args.seed)

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs = RecordEpisodeStatistics(envs)
    envs = gym.wrappers.ClipAction(envs)
    envs = gym.wrappers.NormalizeObservation(envs)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: np.clip(obs, -10, 10))
    envs = gym.wrappers.NormalizeReward(envs)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(action_dim=np.prod(envs.single_action_space.shape))
    actor_parameters = actor.init(jaxRNG, envs.single_observation_space.sample())
    actor.apply = jax.jit(actor.apply)
    critic = Critic()
    critic_parameters = critic.init(jaxRNG, envs.single_observation_space.sample())
    critic.apply = jax.jit(critic.apply)
    actor_optimizer = optax.adam(learning_rate=args.learning_rate, eps=1e-5)
    actor_optimizer_state = actor_optimizer.init((actor_parameters, critic_parameters))

    # ALGO Logic: Storage setup
    obs = jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = jnp.zeros((args.num_steps, args.num_envs))
    rewards = jnp.zeros((args.num_steps, args.num_envs))
    dones = jnp.zeros((args.num_steps, args.num_envs))
    values = jnp.zeros((args.num_steps, args.num_envs))
    avg_returns = deque(maxlen=20)

    @jax.jit
    def get_action_and_value(x, obs, actions, logprobs, values, step, jaxRNG, actor_parameters, critic_parameters):
        obs = obs.at[step].set(x)  # inside jit() `x = x.at[idx].set(y)` is in-place.
        action_mean, action_logstd = actor.apply(actor_parameters, x)
        action_std = jnp.exp(action_logstd)
        action = action_mean + action_std * jax.random.normal(jaxRNG, shape=action_mean.shape)
        logprob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
        value = critic.apply(critic_parameters, x)
        actions = actions.at[step].set(action)
        logprobs = logprobs.at[step].set(logprob.sum(1))
        values = values.at[step].set(value.squeeze())
        return x, obs, actions, logprobs, values, action, logprob, entropy, value

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = np.zeros(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            next_obs, obs, actions, logprobs, values, action, logprob, entropy, value = get_action_and_value(
                next_obs, obs, actions, logprobs, values, step, jaxRNG, actor_parameters, critic_parameters
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(np.array(action))
            for idx, d in enumerate(next_done):
                if d:
                    print(f"global_step={global_step}, episodic_return={info['r'][idx]}")
                    avg_returns.append(info["r"][idx])
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", info["r"][idx], global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        # # bootstrap value if not done
        # with torch.no_grad():
        #     next_value = agent.get_value(next_obs).reshape(1, -1)
        #     if args.gae:
        #         advantages = torch.zeros_like(rewards).to(device)
        #         lastgaelam = 0
        #         for t in reversed(range(args.num_steps)):
        #             if t == args.num_steps - 1:
        #                 nextnonterminal = 1.0 - next_done
        #                 nextvalues = next_value
        #             else:
        #                 nextnonterminal = 1.0 - dones[t + 1]
        #                 nextvalues = values[t + 1]
        #             delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
        #             advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        #         returns = advantages + values
        #     else:
        #         returns = torch.zeros_like(rewards).to(device)
        #         for t in reversed(range(args.num_steps)):
        #             if t == args.num_steps - 1:
        #                 nextnonterminal = 1.0 - next_done
        #                 next_return = next_value
        #             else:
        #                 nextnonterminal = 1.0 - dones[t + 1]
        #                 next_return = returns[t + 1]
        #             returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
        #         advantages = returns - values

        # # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        # # Optimizing the policy and value network
        # b_inds = np.arange(args.batch_size)
        # clipfracs = []
        # for epoch in range(args.update_epochs):
        #     np.random.shuffle(b_inds)
        #     for start in range(0, args.batch_size, args.minibatch_size):
        #         end = start + args.minibatch_size
        #         mb_inds = b_inds[start:end]

        #         _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
        #         logratio = newlogprob - b_logprobs[mb_inds]
        #         ratio = logratio.exp()

        #         with torch.no_grad():
        #             # calculate approx_kl http://joschu.net/blog/kl-approx.html
        #             old_approx_kl = (-logratio).mean()
        #             approx_kl = ((ratio - 1) - logratio).mean()
        #             clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

        #         mb_advantages = b_advantages[mb_inds]
        #         if args.norm_adv:
        #             mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        #         # Policy loss
        #         pg_loss1 = -mb_advantages * ratio
        #         pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        #         pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        #         # Value loss
        #         newvalue = newvalue.view(-1)
        #         if args.clip_vloss:
        #             v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        #             v_clipped = b_values[mb_inds] + torch.clamp(
        #                 newvalue - b_values[mb_inds],
        #                 -args.clip_coef,
        #                 args.clip_coef,
        #             )
        #             v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        #             v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        #             v_loss = 0.5 * v_loss_max.mean()
        #         else:
        #             v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        #         entropy_loss = entropy.mean()
        #         loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        #         optimizer.zero_grad()
        #         loss.backward()
        #         nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        #         optimizer.step()

        #     if args.target_kl is not None:
        #         if approx_kl > args.target_kl:
        #             break

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
