import os
import re
import time
import random
import argparse
import collections
import numpy as np
from distutils.util import strtobool

import torch
import torch as th # shorthand; TODO: clean up once done porting
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Uniform
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import gym
import d4rl
import pybullet_envs
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Box

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conservative Q-Learning with SAC')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Hopper-v2",
                        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--n-epochs', type=int, default=1000,
                        help='number of epochs (total training iters = n-epochs * epoch-length')
    parser.add_argument('--epoch-length', type=int, default=1000,
                        help='number of training steps in an epoch')
    
    # Algorithm specific arguments
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1, # Denis Yarats' implementation delays this by 2.
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256, # Worked better in my experiments, still have to do ablation on this. Please remind me
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    parser.add_argument('--learning-starts', type=int, default=5e3,
                        help="timestep to start learning")

    # SAC specific arguments
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                        help='the learning rate of the optimizer for the policy weights')
    parser.add_argument('--q-lr', type=float, default=1e-3,
                        help='the learning rate of the optimizer for the Q netowrks weights')
    parser.add_argument('--alpha-lr', type=float, default=1e-3,
                        help='the learning rate of the optimizer for the alpha coefficients')
    parser.add_argument('--autotune', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='automatic tuning of the entropy coefficient.')
    parser.add_argument('--policy-frequency', type=int, default=1,
                        help='delays the update of the actor, as per the TD3 paper.')

    # NN Parameterization
    parser.add_argument('--weights-init', default='xavier', const='xavier', nargs='?', choices=['xavier', "orthogonal", 'uniform'],
                        help='weight initialization scheme for the neural networks.')
    parser.add_argument('--bias-init', default='zeros', const='xavier', nargs='?', choices=['zeros', 'uniform'],
                        help='weight initialization scheme for the neural networks.')
    
    # CQL specific parameters
    parser.add_argument('--offline-dataset-id', type=str, default="expert-v0",
                        help='the id of the offline dataset gym environment')
    parser.add_argument('--min-q-weight', type=float, default=5.0,
                        help='coefficient of the lower-bounding term in the CQL loss')
    parser.add_argument('--min-q-version', type=int, default=2,
                        help='type of lower-bounding CQL component used: 2 -> CQL(H), 3 -> CQL(\rho) using current policy')
    parser.add_argument('--num-actions', type=int, default=10,
                        help='number of actions sampled to estimate the CQL lower-bounding term')
    parser.add_argument('--lagrange-thresh', type=float, default=0.0,
                        help='Lagrange threshold for automatic tuning of the alpha_prime component')
    parser.add_argument('--with-lagrange', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='automatic tuning of the entropy coefficient.')
    
    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

    # create offline gym id: 'BeamRiderNoFrameskip-v4' -> 'beam-rider-expert-v0'
    args.offline_gym_id = re.sub(r'(?<!^)(?=[A-Z])', '-', args.gym_id).lower().replace(
        "v2", "") + args.offline_dataset_id

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# ALGO LOGIC: initialize agent here:
LOG_STD_MAX = 2
LOG_STD_MIN = -5

def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        if args.weights_init == "xavier":
            torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        elif args.weights_init == "orthogonal":
            torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
        if args.bias_init == "zeros":
            torch.nn.init.constant_(layer.bias, bias_const)

class Policy(nn.Module):
    def __init__(self, input_shape, output_shape, env):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256) # Better result with slightly wider networks.
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, output_shape)
        self.logstd = nn.Linear(256, output_shape)
        # action rescaling
        self.register_buffer("action_scale", 
            torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.))
        self.register_buffer("action_bias", 
            torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.))
        self.apply(layer_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) +  1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape+output_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.apply(layer_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Offline RL data loading
class ExperienceReplayDataset(Dataset):
    def __init__(self, env_name, device="cpu"):
        self.dataset_env = gym.make(env_name)
        self.dataset = self.dataset_env.get_dataset()
        self.device = device # handles putting the data on the davice when sampling
    def __len__(self):
        return self.dataset['observations'].shape[0]-1
    def __getitem__(self, index):
        return self.dataset['observations'][:-1][index].astype(np.float32), \
               self.dataset['actions'][:-1][index].astype(np.float32), \
               self.dataset['rewards'][:-1][index].astype(np.float32), \
               self.dataset['observations'][1:][index].astype(np.float32), \
               self.dataset['terminals'][:-1][index].astype(np.float32)

data_loader = DataLoader(ExperienceReplayDataset(args.offline_gym_id), batch_size=args.batch_size, num_workers=2)

pg = Policy(input_shape, output_shape, env).to(device)
qf1 = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf2 = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf1_target = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf2_target = SoftQNetwork(input_shape, output_shape, layer_init).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
policy_optimizer = optim.Adam(list(pg.parameters()), lr=args.policy_lr)
loss_fn = nn.MSELoss()

# Automatic entropy tuning for SAC
if args.autotune:
    target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

# Automatic tuming for the min Q term alpha (_prime) coefficient
if args.with_lagrange:
    log_alpha_prime = torch.zeros([1], requires_grad=True, device=device)
    alpha_prime_optimizer = optim.Adam([log_alpha_prime], args.alpha_lr)

# Helper to evaulate the agent
def test_agent(env, policy, n_eval_episodes=5):
    returns, lengths = [], []

    for _ in range(n_eval_episodes):
        ret, done, t = 0., False, 0
        obs = np.array( env.reset())

        while not done:
            # MaxEntRL Paper argues eval should not be determinsitic
            with th.no_grad():
                action, _, _ = policy.get_action(th.Tensor(obs).unsqueeze(0).to(device))
                action = action.tolist()[0]

            obs, rew, done, _ = env.step( action)
            obs = np.array( obs)
            ret += rew
            t += 1

        returns.append(ret)
        lengths.append(t)

    eval_stats = {
        "test_mean_return": np.mean(returns),
        "test_mean_length": np.mean(lengths),
        "test_max_return": np.max(returns),
        "test_max_length": np.max(lengths),
        "test_min_return": np.min(returns),
        "test_min_length": np.min(lengths)
    }

    return eval_stats

# Training Loop
global_step = 0 # Tracks the Gradient updates
for epoch in range(args.n_epochs):
    for k, train_batch in enumerate(data_loader):
        global_step += 1
        # tenosrize, place and the correct device
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = [i.to(device) for i in train_batch]
        
        # Standard Bellman update component
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = pg.get_action(s_next_obses)
            qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions)
            qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = s_rewards + (1 - s_dones) * args.gamma * min_qf_next_target.view(-1)

        qf1_a_values = qf1.forward(s_obs, s_actions).view(-1)
        qf2_a_values = qf2.forward(s_obs, s_actions).view(-1)
        qf1_loss = loss_fn(qf1_a_values, next_q_value)
        qf2_loss = loss_fn(qf2_a_values, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2 # used for stats only

        # CQL
        s_obs_rep = s_obs.repeat_interleave(args.num_actions, 0)
        s_next_obs_rep = s_next_obses.repeat_interleave(args.num_actions, 0)
        random_actions_tensor = torch.zeros([args.batch_size * args.num_actions, output_shape]).uniform_(0., 1.).to(device)
        curr_actions, curr_log_pis, _ = pg.get_action(s_obs_rep)
        next_actions, next_log_pis, _ = pg.get_action(s_next_obs_rep)

        q1_rand, q2_rand = qf1(s_obs_rep, random_actions_tensor).view(args.batch_size, args.num_actions), qf2(s_obs_rep, random_actions_tensor).view(args.batch_size, args.num_actions)
        q1_curr_actions, q2_curr_actions = qf1(s_obs_rep, curr_actions).view(args.batch_size, args.num_actions), qf2(s_obs_rep, curr_actions).view(args.batch_size, args.num_actions)
        q1_next_actions, q2_next_actions = qf1(s_obs_rep, next_actions).view(args.batch_size, args.num_actions), qf2(s_obs_rep, next_actions).view(args.batch_size, args.num_actions)

        if args.min_q_version == 2:
            all_q1 = th.cat([q1_rand, q1_curr_actions, q1_next_actions, qf1_a_values.unsqueeze(-1)], 1)
            all_q2 = th.cat([q2_rand, q2_curr_actions, q2_next_actions, qf2_a_values.unsqueeze(-1)], 1)
        elif args.min_q_version == 3:
            random_density = np.log(0.5 ** random_actions_tensor.shape[-1])
            all_q1 = th.cat([q1_rand - random_density, q1_curr_actions - curr_log_pis.view(args.batch_size, args.num_actions).detach(), q1_next_actions - next_log_pis.view(args.batch_size, args.num_actions).detach()], 1)
            all_q2 = th.cat([q2_rand - random_density, q2_curr_actions - curr_log_pis.view(args.batch_size, args.num_actions).detach(), q2_next_actions - next_log_pis.view(args.batch_size, args.num_actions).detach()], 1)
        else:
            raise NotImplementedError

        min_qf1_loss, min_qf2_loss = th.logsumexp(all_q1, 1), th.logsumexp(all_q2, 1)

        min_qf1_loss = (min_qf1_loss - qf1_a_values).mean() * args.min_q_weight
        min_qf2_loss = (min_qf2_loss - qf2_a_values).mean() * args.min_q_weight

        if args.with_lagrange:
            alpha_prime = th.clamp(log_alpha_prime.exp(), min=0.0, max=10000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - args.lagrange_thresh)
            min_qf2_loss = alpha_prime * (min_qf2_loss - args.lagrange_thresh)

            alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = -.5 * (min_qf1_loss + min_qf2_loss)
            alpha_prime_loss.backward(retain_graph=True)
            alpha_prime_optimizer.step()
        
        cql_qf1_loss = qf1_loss + min_qf1_loss
        cql_qf2_loss = qf2_loss + min_qf2_loss

        cql_qf_loss = (cql_qf1_loss + cql_qf2_loss) / 2.

        values_optimizer.zero_grad()
        cql_qf_loss.backward()
        values_optimizer.step()

        ## Policy network update
        if global_step % args.policy_frequency == 0: # TD 3 Delayed update support
            for _ in range(args.policy_frequency): # compensate for the delay by doing 'actor_update_interval' instead of 1
                pi, log_pi, _ = pg.get_action(s_obs)
                qf1_pi = qf1.forward(s_obs, pi)
                qf2_pi = qf2.forward(s_obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
            
            if args.autotune:
                with torch.no_grad():
                    _, log_pi, _ = pg.get_action(s_obs)
                alpha_loss = ( -log_alpha * (log_pi + target_entropy)).mean()

                a_optimizer.zero_grad()
                alpha_loss.backward()
                a_optimizer.step()
                alpha = log_alpha.exp().item()
            
        # update the target network
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        
        # Stop iteration over the dataset
        if k >= args.epoch_length-1:
            break
    
    eval_stats = test_agent(env, pg)
    print("[E%4d|I%8d] -- PLoss: %.3f -- QLoss: %.3f -- CQLQloss: %.3f --  Train Mean Ret: %.3f" %
        (epoch, global_step, policy_loss.item(), qf_loss.item(), cql_qf_loss.item(),
        eval_stats["test_mean_return"]))
    if args.with_lagrange:
        print("\t APrimeLoss: %.3f" % alpha_prime_loss.item())

    writer.add_scalar("losses/soft_q_value_1_loss", qf1_loss.item(), global_step)
    writer.add_scalar("losses/soft_q_value_2_loss", qf2_loss.item(), global_step)
    writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
    writer.add_scalar("losses/cql_qf_loss", cql_qf_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
    writer.add_scalar("losses/alpha", alpha, global_step)
    writer.add_scalar("charts/episode_reward", eval_stats["test_mean_return"], global_step)
    writer.add_scalar("global_step", global_step, global_step)
    if args.autotune:
        writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
    if args.with_lagrange:
        writer.add_scalar("losses/alpha_prime_loss", alpha_prime_loss.item(), global_step)
        writer.add_scalar("losses/alpha_prime", log_alpha_prime.exp().item(), global_step)

writer.close()
env.close()