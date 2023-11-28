# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppg/#ppg_procgenpy
import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from procgen import ProcgenEnv
from torch import distributions as td
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
    env_id: str = "starpilot"
    """the id of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    adv_norm_fullbatch: bool = True
    """Toggle full batch advantage normalization as used in PPG code"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # PPG specific arguments
    n_iteration: int = 32
    """N_pi: the number of policy update in the policy phase """
    e_policy: int = 1
    """E_pi: the number of policy update in the policy phase """
    v_value: int = 1
    """E_V: the number of policy update in the policy phase """
    e_auxiliary: int = 6
    """E_aux:the K epochs to update the policy"""
    beta_clone: float = 1.0
    """the behavior cloning coefficient"""
    num_aux_rollouts: int = 4
    """the number of mini batch in the auxiliary phase"""
    n_aux_grad_accum: int = 1
    """the number of gradient accumulation in mini batch"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_phases: int = 0
    """the number of phases (computed in runtime)"""
    aux_batch_rollouts: int = 0
    """the number of rollouts in the auxiliary phase (computed in runtime)"""


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = torch.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert torch.equal(a, c)


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale)
        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0)
        nblocks = 2  # Set to the number of residual blocks
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        chans = [16, 32, 32]
        scale = 1 / np.sqrt(len(chans))  # Not fully sure about the logic behind this but its used in PPG code
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        encodertop = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            encodertop,
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)
        self.actor = layer_init_normed(nn.Linear(256, envs.single_action_space.n), norm_dim=1, scale=0.1)
        self.critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        self.aux_critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.detach())

    def get_value(self, x):
        return self.critic(self.network(x.permute((0, 3, 1, 2)) / 255.0))  # "bhwc" -> "bchw"

    # PPG logic:
    def get_pi_value_and_aux_value(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden.detach()), self.aux_critic(hidden)

    def get_pi(self, x):
        return Categorical(logits=self.actor(self.network(x.permute((0, 3, 1, 2)) / 255.0)))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_phases = int(args.num_iterations // args.n_iteration)
    args.aux_batch_rollouts = int(args.num_envs * args.n_iteration)
    assert args.v_value == 1, "Multiple value epoch (v_value != 1) is not supported yet"
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

    flatten_unflatten_test()  # Try not to mess with the flatten unflatten logic

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = ProcgenEnv(num_envs=args.num_envs, env_name=args.env_id, num_levels=0, start_level=0, distribution_mode="easy")
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    aux_obs = torch.zeros(
        (args.num_steps, args.aux_batch_rollouts) + envs.single_observation_space.shape, dtype=torch.uint8
    )  # Saves lot system RAM
    aux_returns = torch.zeros((args.num_steps, args.aux_batch_rollouts))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for phase in range(1, args.num_phases + 1):

        # POLICY PHASE
        for update in range(1, args.n_iteration + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # PPG code does full batch advantage normalization
            if args.adv_norm_fullbatch:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.e_policy):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # PPG Storage - Rollouts are saved without flattening for sampling full rollouts later:
            storage_slice = slice(args.num_envs * (update - 1), args.num_envs * update)
            aux_obs[:, storage_slice] = obs.cpu().clone().to(torch.uint8)
            aux_returns[:, storage_slice] = returns.cpu().clone()

        # AUXILIARY PHASE
        aux_inds = np.arange(args.aux_batch_rollouts)

        # Build the old policy on the aux buffer before distilling to the network
        aux_pi = torch.zeros((args.num_steps, args.aux_batch_rollouts, envs.single_action_space.n))
        for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
            end = start + args.num_aux_rollouts
            aux_minibatch_ind = aux_inds[start:end]
            m_aux_obs = aux_obs[:, aux_minibatch_ind].to(torch.float32).to(device)
            m_obs_shape = m_aux_obs.shape
            m_aux_obs = flatten01(m_aux_obs)
            with torch.no_grad():
                pi_logits = agent.get_pi(m_aux_obs).logits.cpu().clone()
            aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2])
            del m_aux_obs

        for auxiliary_update in range(1, args.e_auxiliary + 1):
            print(f"aux epoch {auxiliary_update}")
            np.random.shuffle(aux_inds)
            for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
                end = start + args.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]
                try:
                    m_aux_obs = aux_obs[:, aux_minibatch_ind].to(device)
                    m_obs_shape = m_aux_obs.shape
                    m_aux_obs = flatten01(m_aux_obs)  # Sample full rollouts for PPG instead of random indexes
                    m_aux_returns = aux_returns[:, aux_minibatch_ind].to(torch.float32).to(device)
                    m_aux_returns = flatten01(m_aux_returns)

                    new_pi, new_values, new_aux_values = agent.get_pi_value_and_aux_value(m_aux_obs)

                    new_values = new_values.view(-1)
                    new_aux_values = new_aux_values.view(-1)
                    old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(device)
                    old_pi = Categorical(logits=old_pi_logits)
                    kl_loss = td.kl_divergence(old_pi, new_pi).mean()

                    real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                    aux_value_loss = 0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                    joint_loss = aux_value_loss + args.beta_clone * kl_loss

                    loss = (joint_loss + real_value_loss) / args.n_aux_grad_accum
                    loss.backward()

                    if (i + 1) % args.n_aux_grad_accum == 0:
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()  # This cannot be outside, else gradients won't accumulate

                except RuntimeError as e:
                    raise Exception(
                        "if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory"
                    ) from e

                del m_aux_obs, m_aux_returns
        writer.add_scalar("losses/aux/kl_loss", kl_loss.mean().item(), global_step)
        writer.add_scalar("losses/aux/aux_value_loss", aux_value_loss.item(), global_step)
        writer.add_scalar("losses/aux/real_value_loss", real_value_loss.item(), global_step)

    envs.close()
    writer.close()
