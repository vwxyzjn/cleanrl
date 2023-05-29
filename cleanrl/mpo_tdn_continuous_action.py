import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Any, Dict, List, NamedTuple, Optional, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from torch.utils.tensorboard import SummaryWriter

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

# MPO with TD(n) as critic loss


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
    parser.add_argument("--headless-server", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether the server to capture videos doesn't have a monitor")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v2",
        help="the id of the environment")
    
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    
    parser.add_argument("--epsilon-non-parametric", type=float, default=0.1,
        help="KL regularization coefficient for the non-parametric distribution")
    parser.add_argument("--epsilon-parametric-mu", type=float, default=0.01,
        help="KL regularization coefficient for the mean of the parametric policy distribution (0.0005 in 2018b paper, 0.0025 in jax deepmind implementation, 0.01 in deepmind example for reproducibility)")
    parser.add_argument("--epsilon-parametric-sigma", type=float, default=1e-6,
        help="KL regularization coefficient for the std of the parametric policy distribution (0.00001 in 2018b paper, 1e-6 in jax deepmind implementation and in deepmind example for reproducibility)")
    parser.add_argument("--epsilon-penalty", type=float, default=0.001,
        help="KL regularization coefficient for the action limit penalty")
    parser.add_argument("--target-network-update-period", type=float, default=100,
        help="number of steps before updating the target networks (250 in 2018b paper, 100 in jax deepmind implementation and in deepmind example for reproducibility)")
    parser.add_argument("--action-sampling-number", type=float, default=20,
        help="number of actions to sample for each state sampled, to compute an approximated non parametric better policy distribution")
    parser.add_argument("--grad-norm-clip", type=float, default=40.,
        help="gradients norm clipping coefficient")
    
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=1e3,
        help="timestep to start learning")
    parser.add_argument("--policy-q-lr", type=float, default=3e-4,
        help="the learning rate of the policy network and critic network optimizer (3e-4 in 2018b paper, 1e-4 in jax deepmind implementation, 3e-4 in deepmind example for reproducibility)")
    parser.add_argument("--dual-lr", type=float, default=1e-2,
        help="the learning rate of the dual parameters")
    
    
    parser.add_argument("--policy-init-scale", type=int, default=0.5,
        help="scaling coefficient of the policy std (not specified in 2018b paper, 0.7 in deepmind jax implementation, 0.5 in deepmind example for reproducibility)")
    parser.add_argument("--policy-min-scale", type=int, default=1e-6,
        help="scalar to add to the scaled std of the policy (not specified in 2018b paper, 1e-6 in jax deepmind implementation and in deepmind example for reproducibility)")
    parser.add_argument("--n_step", type=float, default=4,
        help="horizon for bootstrapping the target q-value (5 in 2018a paper, 5 in jax deepmind implementation, 4 in deepmind example for reproducibility)")
    
    args = parser.parse_args()
    # fmt: on
    return args


_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0, variance_scaling=False):
    if variance_scaling:
        std = torch.sqrt(std / torch.tensor(layer.weight.shape[1]))
        distribution_stddev = torch.tensor(0.87962566103423978)
        std /= distribution_stddev

    torch.nn.init.trunc_normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def uniform_scaling_layer_init(layer, bias_const=0.0, scale=0.333):
    max_val = torch.sqrt(torch.tensor(3.0) / torch.tensor(layer.weight.shape[1])) * scale
    torch.nn.init.uniform_(layer.weight, a=-max_val, b=max_val)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.linear_1 = uniform_scaling_layer_init(nn.Linear(env.single_observation_space.shape[0], 256))
        self.layer_norm = nn.LayerNorm(256)
        self.linear_2 = uniform_scaling_layer_init(nn.Linear(256, 256))
        self.linear_3 = uniform_scaling_layer_init(nn.Linear(256, 256))

        self._loc_layer = layer_init(nn.Linear(256, env.single_action_space.shape[0]), std=1e-4, variance_scaling=True)
        self._scale_layer = layer_init(nn.Linear(256, env.single_action_space.shape[0]), std=1e-4, variance_scaling=True)

    def forward(self, x):
        h = self.linear_1(x)
        h = torch.tanh(self.layer_norm(h))

        h = F.elu(self.linear_2(h))
        h = F.elu(self.linear_3(h))

        loc = self._loc_layer(h)
        scale = F.softplus(self._scale_layer(h))

        scale *= args.policy_init_scale / F.softplus(torch.zeros(1, device=x.device))
        scale += args.policy_min_scale

        return loc, scale


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.value_linear_1 = uniform_scaling_layer_init(
            nn.Linear(env.single_observation_space.shape[0] + env.single_action_space.shape[0], 256)
        )  # 512 in deepmind jax implementation by default, 256 in example
        self.layer_norm = nn.LayerNorm(256)

        self.value_linear_2 = uniform_scaling_layer_init(
            nn.Linear(256, 256)
        )  # 512 in deepmind jax implementation by default, 256 in example
        self.value_linear_3 = uniform_scaling_layer_init(nn.Linear(256, 256))
        self.value_linear_4 = layer_init(nn.Linear(256, 1), std=0.01)

    def forward(self, x, a):
        a = torch.clip(a, -1, 1)
        x = torch.cat([x, a], -1)

        h = self.value_linear_1(x)
        h = torch.tanh(self.layer_norm(h))

        h = F.elu(self.value_linear_2(h))
        h = F.elu(self.value_linear_3(h))

        torch_value = self.value_linear_4(h)

        return torch_value


class TDNReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    bootstrapped_discounts: torch.Tensor


class TDNReplayBuffer(ReplayBuffer):
    r"""
    We extend stable_baseline3 for TD(n) with a new buffer that stores bootstrapped_discount:
    when we add the remaining elements of the n_step rolling buffer after the environment is done,
    we have to specify the int used in exponent to lambda when calculating
    the discount factor that is multiplied with the bootstrapped predicted value by our qfunction.
    It is equal to n_step most of the time, but for the remaining elements in the n_step rolling buffer after
    the environment is done, the horizon isn't n_step anymore, but less, so in these cases it is
    strictly smaller than 0.
    /!\ This is also useful when the episode lasts strictly less than n_step
    /!\ When the episode is done, there is no bootstrapping happening, so one might think this is useless,
    /!\ but the acme introduction paper specifies that in case of episode TRUNCATION bootstrapping should occur,
    /!\ I couldn't verify for sure that this is how they implemented acme, because of the complexity of the modularity
    /!\ of this library, but I take the paper for it.
    /!\ NOTE: stable_baseline3 ReplayBuffer needs the TimeLimit.truncated field of info to be set to True when truncation occurs
    /!\ to handle correctly truncation, but gym doesn't set this field, so we have to do it.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        self.bootstrapped_discounts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage = self.bootstrapped_discounts.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete (augmented with bootstrapped_discounts)"
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        bootstrapped_discount: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        # Important to set before super, because super increases self.pos
        self.bootstrapped_discounts[self.pos] = np.array(bootstrapped_discount).copy()
        super().add(obs=obs, next_obs=next_obs, action=action, reward=reward, done=done, infos=infos)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> TDNReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.bootstrapped_discounts[batch_inds, env_indices].reshape(-1, 1),
        )
        return TDNReplayBufferSamples(*tuple(map(self.to_torch, data)))


if __name__ == "__main__":
    args = parse_args()
    if args.headless_server:
        import pyvirtualdisplay

        # Creates a virtual display for OpenAI gym
        pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
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
    eval_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())

    qf = QNetwork(envs).to(device)
    target_qf = QNetwork(envs).to(device)
    target_qf.load_state_dict(qf.state_dict())

    log_eta = torch.tensor([10.0], requires_grad=True, device=device)

    # Here we only implement per dimension KL constraint
    log_alpha_mean = torch.tensor([10.0] * envs.single_action_space.shape[0], requires_grad=True, device=device)
    log_alpha_stddev = torch.tensor([1000.0] * envs.single_action_space.shape[0], requires_grad=True, device=device)

    # From MO-MPO (but it's not clear why): penalizing actions outside the range
    log_penalty_temperature = torch.tensor([10.0], requires_grad=True, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = TDNReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.policy_q_lr)
    critic_optimizer = torch.optim.Adam(qf.parameters(), lr=args.policy_q_lr)
    dual_optimizer = torch.optim.Adam([log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature], lr=args.dual_lr)

    obs, _ = envs.reset(seed=args.seed)

    n_step_obs_rolling_buffer = np.zeros((args.n_step,) + envs.single_observation_space.shape)
    n_step_action_rolling_buffer = np.zeros((args.n_step,) + envs.single_action_space.shape)
    n_step_reward_rolling_buffer = np.zeros((args.n_step,))
    n_step_gammas = args.gamma ** np.arange(args.n_step)

    step_since_last_done = 0
    sgd_steps = 0

    start_time = time.time()
    for global_step in range(args.total_timesteps):
        with torch.no_grad():
            taus_mean, taus_stddev = actor(torch.Tensor(obs).to(device))
            distribution = torch.distributions.multivariate_normal.MultivariateNormal(
                loc=taus_mean, scale_tril=torch.diag_embed(taus_stddev)
            )
            taus = distribution.sample().cpu()

        next_obs, reward, terminated, truncated, infos = envs.step(taus.numpy().clip(-1, 1))
        done = np.logical_or(terminated, truncated)

        n_step_obs_rolling_buffer = np.concatenate([n_step_obs_rolling_buffer[1:], obs], 0)
        n_step_action_rolling_buffer = np.concatenate([n_step_action_rolling_buffer[1:], taus], 0)
        n_step_reward_rolling_buffer = np.concatenate([n_step_reward_rolling_buffer[1:], reward], 0)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # Problems caused by https://github.com/openai/gym/blob/master/gym/vector/sync_vector_env.py
        real_next_obs = next_obs.copy()

        # It seems that deepmind's acme doesn't use the real last observation
        # when episode is done, but the first observation of the next episode?
        # Here we take another path and do it correctly.
        for idx, d in enumerate(done):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]

        if step_since_last_done >= args.n_step - 1:
            n_step_discounted_reward = (n_step_reward_rolling_buffer * n_step_gammas).sum()
            rb.add(
                n_step_obs_rolling_buffer[0],
                real_next_obs,
                n_step_action_rolling_buffer[0],
                n_step_discounted_reward,
                done,
                np.ones((1,)) * args.n_step,
                [{"TimeLimit.truncated": truncated[0]}],
            )
        else:
            n_step_discounted_reward = (
                n_step_reward_rolling_buffer[args.n_step - 1 - step_since_last_done :]
                * n_step_gammas[: step_since_last_done + 1]
            ).sum()
            rb.add(
                n_step_obs_rolling_buffer[args.n_step - 1 - step_since_last_done],
                real_next_obs,
                n_step_action_rolling_buffer[args.n_step - 1 - step_since_last_done],
                n_step_discounted_reward,
                done,
                np.ones((1,)) * (step_since_last_done + 1),
                [{"TimeLimit.truncated": truncated[0]}],
            )

        step_since_last_done += 1
        obs = next_obs

        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue

                if step_since_last_done >= args.n_step - 1:
                    # Case where rolling_buffer was filled (env ends after n_step)
                    # and therefore we've already dealt with the first entry of the rolling buffer
                    for i in range(1, args.n_step):
                        n_step_discounted_reward = (n_step_reward_rolling_buffer[i:] * n_step_gammas[:-i]).sum()
                        rb.add(
                            n_step_obs_rolling_buffer[i],
                            real_next_obs,
                            n_step_action_rolling_buffer[i],
                            n_step_discounted_reward,
                            done,
                            np.ones((1,)) * (args.n_step - i),
                            [{"TimeLimit.truncated": truncated[0]}],
                        )
                else:
                    # Case where env ends before n_step
                    # First entry wasn't dealt with
                    for i in range(0, step_since_last_done):
                        n_step_discounted_reward = (n_step_reward_rolling_buffer[i:] * n_step_gammas[:-i]).sum()
                        rb.add(
                            n_step_obs_rolling_buffer[i],
                            real_next_obs,
                            n_step_action_rolling_buffer[i],
                            n_step_discounted_reward,
                            done,
                            np.ones((1,)) * (step_since_last_done - i),
                            [{"TimeLimit.truncated": truncated[0]}],
                        )

                step_since_last_done = 0
                n_step_obs_rolling_buffer = np.zeros((args.n_step,) + envs.single_observation_space.shape)
                n_step_action_rolling_buffer = np.zeros((args.n_step,) + envs.single_action_space.shape)
                n_step_reward_rolling_buffer = np.zeros((args.n_step,))

                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        if global_step > args.learning_starts:
            if global_step % 4 == 0:
                # PHASE 1
                # QValue learning
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    torch_target_mus, torch_target_stddevs = target_actor(data.next_observations)
                    distribution = torch.distributions.MultivariateNormal(
                        loc=torch_target_mus, scale_tril=torch.diag_embed(torch_target_stddevs)
                    )
                    torch_taus = distribution.sample(torch.Size([args.action_sampling_number]))  # (N, B, A)
                    completed_target_states = data.next_observations.repeat([args.action_sampling_number, 1, 1])

                    target_qvalue = target_qf(completed_target_states, torch_taus).squeeze(-1)  # (N,B)

                    target_qvalue = target_qvalue.mean(0)  # (B,)

                    target_qvalue = (
                        (1 - data.dones.flatten()) * (args.gamma ** (data.bootstrapped_discounts.squeeze(-1))) * target_qvalue
                    )
                    td_qtarget = data.rewards.flatten() + target_qvalue

                old_qval = qf(data.observations, data.actions).squeeze(-1)

                qvalue_loss = 0.5 * torch.square(td_qtarget - old_qval)  # F.mse_loss(td_qtarget, old_qval)
                qvalue_loss = qvalue_loss.mean()

                # N: number of actions sampled
                # B: batch size of states
                # A: number of independent actions
                # PHASE 2
                # Compute improved non-parametric distribution
                # Sample impr_distr_action_nb actions for each state from target actor
                eta = F.softplus(log_eta) + _MPO_FLOAT_EPSILON

                stacked_observations = torch.cat([data.observations, data.next_observations], dim=0)

                with torch.no_grad():
                    target_mean, target_std = target_actor(stacked_observations)
                    target_pred_distribution = torch.distributions.MultivariateNormal(
                        loc=target_mean, scale_tril=torch.diag_embed(target_std)
                    )

                    target_pred_distribution_per_dim_constraining = torch.distributions.Independent(
                        torch.distributions.Normal(loc=target_mean, scale=target_std), reinterpreted_batch_ndims=1
                    )

                    target_sampl_actions = target_pred_distribution.sample(
                        torch.Size([args.action_sampling_number])
                    )  # (N,B,A)

                # Compute their Q-values with the target model
                with torch.no_grad():
                    completed_states = stacked_observations.repeat([args.action_sampling_number, 1, 1])
                    online_q_values_sampl_actions = target_qf(completed_states, target_sampl_actions).squeeze(-1)  # (N, B)

                # Compute new distribution
                impr_distr = F.softmax(online_q_values_sampl_actions / eta.detach(), dim=0)  # shape (N,B)

                # Compute eta loss: optimization of the normalization and KL regularized constraints
                q_logsumexp = torch.logsumexp(online_q_values_sampl_actions / eta, dim=0)  # (B,)
                log_num_actions = torch.log(torch.tensor(args.action_sampling_number))
                loss_eta = args.epsilon_non_parametric + torch.mean(q_logsumexp, dim=0) - log_num_actions
                loss_eta = eta * loss_eta

                # 2020 MO-MPO action range limit penalization
                penalty_temperature = F.softplus(log_penalty_temperature) + _MPO_FLOAT_EPSILON
                diff_out_of_bound = target_sampl_actions - torch.clip(target_sampl_actions, -1, 1)  # (N,B,A)
                cost_out_of_bound = -torch.linalg.norm(diff_out_of_bound, dim=-1)  # (N,B)
                # Compute penalty distribution
                penalty_impr_distr = F.softmax(cost_out_of_bound / penalty_temperature.detach(), dim=0)  # shape (N,B)
                # Compute penalization temperature loss: optimization of the normalization and KL regularized constraints
                panalty_q_logsumexp = torch.logsumexp(cost_out_of_bound / penalty_temperature, dim=0)  # (B,)
                penalty_log_num_actions = torch.log(torch.tensor(args.action_sampling_number))
                loss_penalty_temperature = (
                    args.epsilon_penalty + torch.mean(panalty_q_logsumexp, dim=0) - penalty_log_num_actions
                )
                loss_penalty_temperature = penalty_temperature * loss_penalty_temperature

                impr_distr += penalty_impr_distr
                loss_eta += loss_penalty_temperature

                # PHASE 3
                # Regression on the actions sampled of the online actor
                # to the non-parametric improved distributions
                # Sample from online actor
                alpha_mean = F.softplus(log_alpha_mean) + _MPO_FLOAT_EPSILON
                alpha_stddev = (
                    torch.logaddexp(log_alpha_stddev, torch.tensor(0, device=device)) + _MPO_FLOAT_EPSILON
                )  # F.softplus(log_alpha_stddev) + _MPO_FLOAT_EPSILON

                online_mean, online_std = actor(stacked_observations)

                # Decouple optimization between mean and std
                # Here we begin with mean (we optimize the mean but fixed the std)
                online_pred_distribution_mean = torch.distributions.Independent(
                    torch.distributions.Normal(loc=online_mean, scale=target_std), reinterpreted_batch_ndims=1
                )
                # Compute cross entropy loss
                online_log_probs_mean = online_pred_distribution_mean.log_prob(target_sampl_actions)  # (N,B)

                loss_policy_gradient_mean = -torch.sum(online_log_probs_mean * impr_distr, dim=0)  # (B,)
                loss_policy_gradient_mean = loss_policy_gradient_mean.mean()  # ()

                # Optimization of the KL trust-region constraint
                kl_mean = torch.distributions.kl.kl_divergence(
                    target_pred_distribution_per_dim_constraining.base_dist, online_pred_distribution_mean.base_dist
                )  # (B,A)
                mean_kl_mean = torch.mean(kl_mean, dim=0)  # (A,)
                loss_kl_mean = torch.sum(alpha_mean.detach() * kl_mean, 1).mean()
                loss_alpha_mean = torch.sum(alpha_mean * (args.epsilon_parametric_mu - kl_mean.detach()), 1).mean()

                # Here finish with std (we optimize the std but fixed the mean)
                online_pred_distribution_stddev = torch.distributions.Independent(
                    torch.distributions.Normal(loc=target_mean, scale=online_std), reinterpreted_batch_ndims=1
                )
                # Compute cross entropy loss
                online_log_probs_stddev = online_pred_distribution_stddev.log_prob(target_sampl_actions)  # (N,B)

                loss_policy_gradient_stddev = -torch.sum(online_log_probs_stddev * impr_distr, dim=0)  # (B,)
                loss_policy_gradient_stddev = loss_policy_gradient_stddev.mean()  # ()

                # Optimization of the KL trust-region constraint
                kl_stddev = torch.distributions.kl.kl_divergence(
                    target_pred_distribution_per_dim_constraining.base_dist, online_pred_distribution_stddev.base_dist
                )  # (B,A)
                mean_kl_stddev = torch.mean(kl_stddev, dim=0)  # (A,)
                loss_kl_stddev = torch.sum(alpha_stddev.detach() * kl_stddev, 1).mean()
                loss_alpha_stddev = torch.sum(alpha_stddev * (args.epsilon_parametric_sigma - kl_stddev.detach()), 1).mean()

                actor_loss = loss_policy_gradient_mean + loss_policy_gradient_stddev + loss_kl_mean + loss_kl_stddev
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                qvalue_loss.backward()
                critic_optimizer.step()

                dual_loss = loss_alpha_mean + loss_alpha_stddev + loss_eta
                dual_optimizer.zero_grad()
                dual_loss.backward()
                dual_optimizer.step()

                # The following is a try to do exactly what's implemented in the official deepmind's implementation
                # where they clip the parameters outside the backpropagation algorithm
                log_eta.data.clamp_(min=_MIN_LOG_TEMPERATURE)
                log_alpha_mean.data.clamp_(min=_MIN_LOG_ALPHA)
                log_alpha_stddev.data.clamp_(min=_MIN_LOG_ALPHA)

                sgd_steps += 1

                if sgd_steps % args.target_network_update_period == 0:
                    target_actor.load_state_dict(actor.state_dict())
                    target_qf.load_state_dict(qf.state_dict())

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if sgd_steps % 25 == 0:
                    # global losses and qf values
                    writer.add_scalar("losses/qf_values", old_qval.mean().item(), global_step)
                    writer.add_scalar("losses/qf_loss", qvalue_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("losses/dual_loss", dual_loss.item(), global_step)

                    # dual values
                    writer.add_scalar("losses/log_eta", log_eta.item(), global_step)
                    writer.add_scalar("losses/log_penalty_temperature", log_penalty_temperature.item(), global_step)
                    writer.add_scalar("losses/mean_log_alpha_mean", log_alpha_mean.mean().item(), global_step)
                    writer.add_scalar("losses/mean_log_alpha_stddev", log_alpha_stddev.mean().item(), global_step)

                    # dual losses
                    writer.add_scalar("losses/loss_alpha", (loss_alpha_mean + loss_alpha_stddev).item(), global_step)
                    writer.add_scalar("losses/loss_eta", loss_eta.item(), global_step)

                    # kl values
                    writer.add_scalar(
                        "losses/kl_mean_rel", (mean_kl_mean / args.epsilon_parametric_mu).mean().item(), global_step
                    )
                    writer.add_scalar(
                        "losses/kl_stddev_rel", (mean_kl_stddev / args.epsilon_parametric_sigma).mean().item(), global_step
                    )

                    # stddev of online policy
                    writer.add_scalar("policy/pi_stddev_min", online_std.min(dim=1).values.mean().item(), global_step)
                    writer.add_scalar("policy/pi_stddev_max", online_std.max(dim=1).values.mean().item(), global_step)

                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        eval_every = 5000
        eval_nb = 10
        eval_obs, _ = eval_envs.reset(seed=args.seed)
        eval_episodic_return = np.zeros((eval_nb,))
        eval_episodic_length = np.zeros((eval_nb,))
        if (global_step + 1) % eval_every == 0:
            for e in range(eval_nb):
                eval_done = False
                while not eval_done:
                    with torch.no_grad():
                        taus, _ = actor(torch.Tensor(eval_obs).to(device))
                        taus = taus.cpu()

                    eval_obs, _, eval_terminated, eval_truncated, eval_infos = eval_envs.step(
                        taus.numpy().clip(-1, 1)
                    )
                    eval_done = np.logical_or(eval_terminated, eval_truncated)

                    if "final_info" in eval_infos:
                        for eval_info in eval_infos["final_info"]:
                            # Skip the envs that are not done
                            if eval_info is None:
                                continue

                            print(f"eval={e}, episodic_return={eval_info['episode']['r']}")
                            eval_episodic_return[e] = eval_info["episode"]["r"]
                            eval_episodic_length[e] = eval_info["episode"]["l"]
            writer.add_scalar("evaluation/episodic_return", eval_episodic_return.mean(), global_step)
            writer.add_scalar("evaluation/episodic_length", eval_episodic_length.mean(), global_step)
