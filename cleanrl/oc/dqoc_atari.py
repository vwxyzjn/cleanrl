# Implementation of original option critic
# Reference implementation: https://github.com/jeanharb/option_critic
# DQN-style option-critic for atari
# TODO: Slow. Can we do PG updates on the rollouts at the same cadence as critic updates?
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Any, Dict, Final, List, NamedTuple, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
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
    parser.add_argument("--env-id", type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(1e7),
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--clip-delta", type=float, default=1.,
        help="max absolute value for Q-update delta value")
    parser.add_argument("--target-network-frequency", type=int, default=10000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.1,
        help="the ending epsilon for exploration")
    parser.add_argument("--test-e", type=float, default=0.05,
        help="the epsilon to use when testing")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,  # 1M steps decay
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=50000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--num-options", type=int, default=8,
        help="the number of options available")
    parser.add_argument("--term-reg", type=float, default=0.01,
        help="regularization term added to option advantage to encourage longer options")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy coefficient for intra-option policies")
    args = parser.parse_args()
    # fmt: on
    return args


class ReplayBuffer(ReplayBuffer):
    """Modified to work with newer gym version"""

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Same, for actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = info.get("TimeLimit.truncated", False)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


@torch.jit.script
def batched_index(idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return contents of t at n-D array idx. Leading dim of t must match dims of idx"""
    dim = len(idx.shape)
    assert idx.shape == t.shape[:dim]
    num = idx.numel()
    t_flat = t.view((num,) + t.shape[dim:])
    s_flat = t_flat[torch.arange(num, device=t.device), idx.view(-1)]
    return s_flat.view(t.shape[:dim] + t.shape[dim + 1 :])


def layer_init(
    layer: nn.Module,
    std: float = nn.init.calculate_gain("relu"),
    bias_const: float = 0.0,
):
    """Orthogonal layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Base Nature CNN
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(inplace=True),
        )
        self.critic = layer_init(nn.Linear(512, args.num_options), std=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return processed features (no gradient) and critic output Q(s,w)"""
        x = self.network(x.float() / 255.0)
        return x.detach(), self.critic(x)


class ActorOutput(NamedTuple):
    termination: torch.Tensor
    beta: torch.Tensor
    option: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    entropy: torch.Tensor


class ActorHeads(nn.Module):
    num_actions: Final[int]
    num_options: Final[int]

    def __init__(self, envs, args):
        super().__init__()
        self.num_options = args.num_options
        self.num_actions = envs.single_action_space.n
        self.termination = layer_init(nn.Linear(512, args.num_options), std=0.01)
        self.option_actor = layer_init(nn.Linear(512, int(envs.single_action_space.n * args.num_options)), std=0.01)

    def forward(
        self,
        detached_features: torch.Tensor,
        q: torch.Tensor,
        option_on_arrival: torch.Tensor,
        current_epsilon: float = 1.0,
        random_action_phase: bool = False,
    ) -> ActorOutput:
        """Sample termination. Sample new option if terminal. Sample new action from option"""
        beta = torch.sigmoid(
            batched_index(option_on_arrival, self.termination(detached_features))
        )  # Termination probability of current option
        if random_action_phase:
            termination = torch.ones_like(option_on_arrival, dtype=torch.bool)  # Always terminate in random action phase
        else:
            termination = torch.bernoulli(beta)  # Sample termination
        # Epsilon-greedy option when terminal
        if termination.any():
            candidate_option = torch.where(
                torch.rand_like(option_on_arrival, dtype=torch.float32) < current_epsilon,
                torch.randint_like(option_on_arrival, 0, self.num_options),
                q.argmax(-1),
            )
            option = torch.where(termination > 0, candidate_option, option_on_arrival)
        else:
            option = option_on_arrival
        pi_logits = self.option_actor(detached_features).reshape(-1, self.num_options, self.num_actions)
        logprobs = torch.log_softmax(batched_index(option, pi_logits), -1)  # Log prob for all actions under each option
        probs = logprobs.exp()
        entropy = (-(logprobs * probs)).sum(-1)  # Entropy of distributions
        action = torch.multinomial(probs, 1)
        logprob = logprobs.gather(-1, action).squeeze(-1)
        return ActorOutput(termination, beta, option, action.squeeze(-1), logprob, entropy)

    @torch.jit.export
    def term_probs(self, next_detached_features: torch.Tensor, options_on_arrival: torch.Tensor):
        """No sampling, just get term probs for options at bootstrap step"""
        return batched_index(options_on_arrival.squeeze(-1), self.termination(next_detached_features))


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear decay from start to end over time"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def make_env(env_id):
    def thunk():
        env = gym.make("ALE/" + env_id, frameskip=4, repeat_action_probability=0.0)
        env = gym.wrappers.AtariPreprocessing(env, frame_skip=1)
        env = gym.wrappers.FrameStack(env, 4)  # Still need frame stacking
        return env

    return thunk


@torch.jit.script
def compute_td_target(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_target_qs: torch.Tensor,
    next_termprobs: torch.Tensor,
    options_on_arrival: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    with torch.no_grad():
        non_term = (1.0 - next_termprobs) * batched_index(options_on_arrival, next_target_qs)
        term = next_termprobs * next_target_qs.max(-1)[0]
        return rewards + (1.0 - dones) * gamma * (non_term + term)


@torch.jit.script
def actor_loss(
    oc_output: ActorOutput,
    td_target: torch.Tensor,
    detached_q: torch.Tensor,
    current_option: torch.Tensor,
    term_reg: float = 0.01,
    ent_coef: float = 0.01,
):
    with torch.no_grad():
        q_s = detached_q
        v_s = q_s.max(-1)[0]
        q_sw = batched_index(current_option, q_s)
    pg_loss = (-oc_output.logprob * (td_target - q_sw)).mean()  # PG with baseline
    entropy_loss = (-oc_output.entropy).mean()  # entropy loss
    term_loss = (oc_output.beta * (q_sw - v_s + term_reg)).mean()  # termination loss
    actor_loss = pg_loss + term_loss + ent_coef * entropy_loss
    return actor_loss


critic_loss = torch.jit.script(F.mse_loss)

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

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id)])
    envs = gym.wrappers.RecordEpisodeStatistics(envs)  # must be outside to avoid losing statistics to autoreset
    envs = gym.wrappers.TransformReward(envs, partial(np.clip, a_min=-1.0, a_max=1.0))  # reward clipping after stats
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = torch.jit.script(QNetwork(args).to(device))
    q_optim = optim.RMSprop(q_network.parameters(), lr=args.learning_rate, alpha=0.95, eps=1e-2)
    target_q_network = torch.jit.script(QNetwork(args).to(device))
    target_q_network.load_state_dict(q_network.state_dict())
    oc_head = torch.jit.script(ActorHeads(envs, args).to(device))
    oc_optim = optim.SGD(oc_head.parameters(), lr=args.learning_rate)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        gym.spaces.Discrete(args.num_options),
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    current_option = torch.randint(0, args.num_options, (envs.num_envs,), dtype=torch.int64, device=device)

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put
        # action logic here
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        detached_features, q = q_network(torch.tensor(obs, device=device))
        oc_output = oc_head(
            detached_features,
            q.detach(),
            current_option,
            epsilon,
            global_step <= args.learning_starts,
        )

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, truncateds, info = envs.step(oc_output.action.cpu().numpy())
        dones = np.array(dones) | np.array(truncateds)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in info.keys():
            print(f"global_step={global_step}, episodic_return={info['episode']['r'][0]}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"][0], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"][0], global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

        # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        if dones.any():
            real_next_obs[dones] = info["final_observation"][0][0]
        rb.add(obs, real_next_obs, oc_output.option.cpu(), rewards, dones, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # online actor update. Need target Q for next obs
            with torch.no_grad():
                s, next_target_q = target_q_network(torch.tensor(obs, device=device))
                next_oc_out = oc_head(s, next_target_q, current_option, epsilon, True)
                td_target = compute_td_target(
                    torch.tensor(rewards, device=device),
                    torch.tensor(dones, device=device, dtype=torch.float32),
                    next_target_q,
                    next_oc_out.beta,
                    current_option,
                    args.gamma,
                )
            oc_loss = actor_loss(
                oc_output,
                td_target,
                q.detach(),
                current_option,
                args.term_reg,
                args.ent_coef,
            )
            oc_optim.zero_grad(True)
            oc_gnorm = nn.utils.clip_grad_norm_(q_network.parameters(), args.clip_delta)
            oc_loss.backward()
            oc_optim.step()

            # offline critic update
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_detached_features, next_target_q = target_q_network(data.next_observations)
                    next_termprobs = oc_head.term_probs(next_detached_features, data.actions)
                    td_target = compute_td_target(
                        data.rewards.flatten(),
                        data.dones.flatten(),
                        next_target_q,
                        next_termprobs,
                        data.actions.flatten(),
                        args.gamma,
                    )
                old_val = batched_index(data.actions.flatten(), q_network(data.observations)[-1])
                q_loss = critic_loss(td_target, old_val)

                # optimize the model
                q_optim.zero_grad(True)
                q_gnorm = nn.utils.clip_grad_norm_(q_network.parameters(), args.clip_delta)
                q_loss.backward()
                q_optim.step()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/v_loss", q_loss.item(), global_step)
                    writer.add_scalar("losses/v_grad_norm", q_gnorm.item(), global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("losses/policy_loss", oc_loss.item(), global_step)
                    writer.add_scalar("losses/policy_grad_norm", oc_gnorm.item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_q_network.load_state_dict(q_network.state_dict())

    envs.close()
    writer.close()
