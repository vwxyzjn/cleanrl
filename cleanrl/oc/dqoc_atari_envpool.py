# Implementation of original option critic
# Reference implementation: https://github.com/jeanharb/option_critic
# DQN-style option-critic for atari


# 4-framestack
# Learning rate for intra-option policies and termination is 2.5e-4
# Intra-option policies are linear softmax, termination are sigmoid
# Critic trained using intra-option Q-learning with experience replay. Option/termination policies updated online
# epislon-greedy policy over options, epsilon = 0.05 during test phase
# Small termination regularizer added to advantage function A(s,w) = Q(s,w) - V(s) + 0.01
# Apply entropy regularization to intra-option policies
# Add baseline Q(s,w) to intra-option PG estimator to reduce variance
# RMSProp (0.95 rho, .01 epsilon)

# Environments: Asterisk, Ms. Pacman, Seaquest, Zaxxon
# 8 options

# to train: python train_q.py --rom pong --num-options 8 --folder-name pong_tempmodel
# train_q: Defines defaults, calls launcher.launch(args, defaults, __doc__)
# 250K steps per epoch, 8000 epochs, 130K steps per evaluation
# cap reward,
# launcher.launch():
#   process_args(): Add args to defaults
#   get ROM path, set rng. Get ALEInterface()
#   trainer = Q_learning(model_params=args, ale_env=ale,...)
#   trainer.train()
# train_agent.Q_Learning(). Q_Learning(DQN_Trainer), DQN_Trainer(Trainer)
#   Trainer:
#       init time, save args, setup ALE, 18K max episode steps
#       save minimal action set
#       cap_reward(): Cap reward to 1,-1
#       _init_ep(): Start with a bunch of no-ops
#       act(action): Act in environment, get obs
#       get_observation(): Get grayscale single frame
#       get_epsilon(): Get epsilon, decaying from 1 to 0.1 starting from replay_start
#       run_testing(epoch): run_training_episode(max_frame, testing=True), save results (mean_reward, mean_q). Don't update frame count
#       train():
#           for e in epochs:
#               for step in steps_per_epoch: run_training_episode(max_frames), reset_game
#               self.update_term_probs(i, self.term_probs)
#               self.run_testing(i)
#   DQNTrainer():
#       run_training_episode()
#           get_new_frame(): Get stacked frame
#           data_set: Choose either test replay buffer or experience replay buffer
#           x = _init_ep, s = self.model.get_state(x)  # Processed
#           current_option = 0, current_action = 0, termination=True
#           new_option = self.model.predict_move(s)
#           while not game_over:
#               get_epsilon() or get_testing_epsilon()
#               if termination: current_option = new_option (or random) eps-greedy
#               current_action = self.model.get_action(s, current_option)
#               reward, raw_reward, new_frame = self.act(current_action, testing=testing)
#               data_set.add_sample(x[-1], current_option, reward, game_over or life_death)
#               old_s = copy(s), x = get_new_frame(new_frame, x)  to stack
#               term_out = self.model.predict_termination(s, current_option). termination, new_option = term)out
#               If we're before replay start size, always terminate
#               if we're past replay_start size
#                   self.learn_actor(old_s, x, current_option, current_action, reward, d)  # Every step update actor
#                   if interval: self.learn_critic()  # Use replay buffer to update critic
#                   if freeze_interval: self.model.update_target_params()
#       learn_actor(s, next_x, o, a, r, term): return self.model.train_conv_net(s, next_x, o, r, term, actions=a, model="actor")  # Returns td_errors
#       learn_critic(): Sample x, o, r, next-x, term. self.model.train_conv_net(x, next_x, o, r, term, model="critic")  # Returns tr_errors
#   Q_Learning():
#       OPtionCritic_Network(net_arch, RMSProp, DNN, clip_delta, input_size, batch_size, gamma, freeze_interval, termination_reg,...)
#       self.exp_replay, self.test_replay are Replay buffers. Test has max_steps=4)
# OptionCritic_Network():
#   placeholders for x, next-x, a, o, r, terminal
#   state_network is all arch layers except last (which is num_actions)
#   Dynamically create heads from last layer (term, option, Q). E.g., Q_network is linaer of size num_options
#   self.state_model, self.state_model_prime = Model(state_network)
#   self.q_model, self.q_model_prime = Model(q_network)
#   self.termination_model, self.options_model = Model(termination), MLP3d(num_options
#   s, next_s, next_s_prime = self.state_model.apply(next_x/255)
#   term_probs = self.termination_model.apply(s.detach()), next_term_probs, sample is bernoulli
#   Q = self.Q_model.apply(s), next_Q is next_s, next_q_prime is without grad
#   actions = self.options_model.apply(s, o)
#   y = r + (1-d) * gamma * (1-next_beta) * next_q_prime[w] + beta * max next_Q_prime
#   td_errors = y - Q[w]
#   clip td_error (if clip_delta)
#   td_cost = 0.5 * td_errors ** 2 (MSE)
#   Sum critic cost, use RMSProp to do loss (grad clip as well)
#   Use SGD for actor and termination
#   term_grad = beta * (Q[w] - V + reg)
#   policy_grad: -lp(a|s) * G - Q(s, w)
#   train_conv_net(): # Set placeholders to next_x, option, r, d. If critic, self.train_critic()

import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Final

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--env-id", type=str, default="Pong-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=int(2e9),  # 250K steps per epoch, 8000 epochs
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
        help="total timesteps of the experiments")
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
        help="the number of options avaialble")
    parser.add_argument("--term-reg", type=float, default=0.01,
        help="regularization term added to option advantage to encourage longer options")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy coefficient for intra-option policies")
    args = parser.parse_args()
    # fmt: on
    return args


class RecordEpisodeStatistics(gym.Wrapper):
    """Envpool-compatible episode statistics recording"""
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


def layer_init(layer: nn.Module, std: float = nn.init.calculate_gain('relu'), bias_const: float = 0.):
    """Orthogonal layer initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    num_options: Final[int]
    num_actions: Final[int]
    def __init__(self, envs, args):
        super().__init__()
        # Base DQN
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
        self.num_options = args.num_options
        self.num_actions = envs.single_action_space.n
        self.termination = layer_init(nn.Linear(512, args.num_options), std=0.01)
        self.option_actor = layer_init(nn.Linear(512, int(envs.single_action_space.n * args.num_options)), std=0.01)
        self.critic = layer_init(nn.Linear(512, args.num_options), std=1.)

    def forward(self, x: torch.Tensor, current_option: torch.Tensor, current_epsilon: float = 1.):
        """Return (termination, next option, term_probs), (log_pi, a), q"""
        x = self.network(x / 255.)
        B = x.shape[-2]  # batch size for eps-greedy
        # Detach termination and option actors
        beta = torch.sigmoid(self.termination(x.detach()))
        term = torch.bernoulli(torch.take_along_dim(beta, current_option, -1))
        pi = self.option_actor(x.detach()).reshape(-1, self.num_options, self.num_actions)
        log_pi_w = torch.log_softmax(torch.take_along_dim(pi, current_option, -2), -1)
        a = torch.multinomial(log_pi_w.exp(), 1)
        q = self.critic(x)
        new_option = current_option
        if term:
            new_option = torch.where(torch.rand(B) < current_epsilon,
                                     torch.randint(0, self.num_options, B),
                                     q.argmax(-1))
        return (term, new_option, beta), (log_pi_w.gather(-1, a).squeeze(-1), a), q



def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear decay from start to end over time"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
    # has 4framestack, 4 frameskip, reward clip
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    oc_network = Agent(envs, args).to(device)
    optimizer = optim.RMSprop(oc_network.parameters(), lr=args.learning_rate, alpha=0.95, eps=1e-2)
    target_network = Agent(envs, args).to(device)
    target_network.load_state_dict(oc_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    current_option = torch.randint(0, args.num_options, args.num_envs, dtype=torch.int64)

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = oc_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = oc_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step)
                writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(oc_network.state_dict())

    envs.close()
    writer.close()