import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import distributions as td
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from procgen import ProcgenEnv
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize, VecVideoRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="starpilot",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1e8,
                        help='total timesteps of the experiments')
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

    # Algorithm specific arguments
    parser.add_argument('--n-iteration', type=int, default=32,
                        help="N_pi: the number of policy update in the policy phase ")
    parser.add_argument('--e-policy', type=int, default=1,
                        help="E_pi: the number of policy update in the policy phase ")
    parser.add_argument('--v-value', type=int, default=1,
                        help="E_V: the number of policy update in the policy phase ")
    parser.add_argument('--e-auxiliary', type=int, default=6,
                        help="E_aux:the K epochs to update the policy")
    parser.add_argument('--beta-clone', type=float, default=1.0,
                        help='the behavior cloning coefficient')
    parser.add_argument('--n-aux-minibatch', type=int, default=16,
                        help='the number of mini batch in the auxiliary phase')
    parser.add_argument('--n-aux-grad-accum', type=int, default=10,
                        help='the number of gradient accumulation in mini batch')

    parser.add_argument('--n-minibatch', type=int, default=8,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=64,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)
args.aux_batch_size = int(args.batch_size * args.n_iteration)
args.aux_minibatch_size  = int(args.aux_batch_size // (args.n_aux_minibatch * args.n_aux_grad_accum))

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

class VecExtractDictObs(VecEnvWrapper):
	def __init__(self, venv, key):
	    self.key = key
	    super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])
	
	def reset(self):
	    obs = self.venv.reset()
	    return obs[self.key]
	
	def step_wait(self):
	    obs, reward, done, info = self.venv.step_wait()
	    return obs[self.key], reward, done, info

class VecMonitor(VecEnvWrapper):
	def __init__(self, venv):
	    VecEnvWrapper.__init__(self, venv)
	    self.eprets = None
	    self.eplens = None
	    self.epcount = 0
	    self.tstart = time.time()
	
	def reset(self):
	    obs = self.venv.reset()
	    self.eprets = np.zeros(self.num_envs, 'f')
	    self.eplens = np.zeros(self.num_envs, 'i')
	    return obs
	
	def step_wait(self):
	    obs, rews, dones, infos = self.venv.step_wait()
	    self.eprets += rews
	    self.eplens += 1
	
	    newinfos = list(infos[:])
	    for i in range(len(dones)):
	        if dones[i]:
	            info = infos[i].copy()
	            ret = self.eprets[i]
	            eplen = self.eplens[i]
	            epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
	            info['episode'] = epinfo
	            self.epcount += 1
	            self.eprets[i] = 0
	            self.eplens[i] = 0
	            newinfos[i] = info
	    return obs, rews, dones, newinfos

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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
venv = ProcgenEnv(num_envs=args.num_envs, env_name=args.gym_id, num_levels=0, start_level=0, distribution_mode='hard')
venv = VecExtractDictObs(venv, "rgb")
venv = VecMonitor(venv=venv)
envs = VecNormalize(venv=venv, norm_obs=False)
envs = VecPyTorch(envs, device)
if args.capture_video:
	envs = VecVideoRecorder(envs, f'videos/{experiment_name}', record_video_trigger=lambda x: x % 1000000== 0, video_length=600)
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

# ALGO LOGIC: initialize agent here:
class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, channels=3):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 4, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(8*8*32, 512)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.aux_critic = layer_init(nn.Linear(512, 1), std=1)

        self.critic = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 4, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(8*8*32, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1)
        )

    def get_action(self, x, action=None):
        logits = self.actor(self.network(x.permute((0, 3, 1, 2)))) # "bhwc" -> "bchw" # "bhwc" -> "bchw"
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(x.permute((0, 3, 1, 2)))

    # PPG logic:
    def get_pi(self, x):
        logits = self.actor(self.network(x.permute((0, 3, 1, 2))))
        return Categorical(logits=logits)

    def get_aux_value(self, x):
        return self.aux_critic(self.network(x.permute((0, 3, 1, 2))))

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
aux_obs = torch.zeros((args.num_steps * args.num_envs * args.n_iteration,) + envs.observation_space.shape)
aux_returns = torch.zeros((args.num_steps * args.num_envs * args.n_iteration,))

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = int(args.total_timesteps // args.batch_size)
num_phases = int(num_updates // args.n_iteration)

## CRASH AND RESUME LOGIC:
starting_phase = 1
for phase in range(starting_phase, num_phases):
    for policy_update in range(1, args.n_iteration+1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (policy_update - 1.0) / num_updates
            lrnow = lr(frac)
            optimizer.param_groups[0]['lr'] = lrnow
    
        # TRY NOT TO MODIFY: prepare the execution of the game.
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
    
            # ALGO LOGIC: put action logic here
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                action, logproba, _ = agent.get_action(obs[step])
            actions[step] = action
            logprobs[step] = logproba
    
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rs, ds, infos = envs.step(action)
            rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)
    
            for info in infos:
                if 'episode' in info.keys():
                    print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                    break
    
        # bootstrap reward if not done. reached the batch limit
        with torch.no_grad():
            last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        nextvalues = values[t+1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t+1]
                        next_return = returns[t+1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values
    
        # flatten the batch
        b_obs = obs.reshape((-1,)+envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,)+envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
    
        # Optimizaing the policy and value network
        inds = np.arange(args.batch_size,)
        for i_epoch_pi in range(args.e_policy):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]
                mb_advantages = b_advantages[minibatch_ind]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
    
                _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()
    
                # Stats
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()
    
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
    
                # Value loss
                new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                    v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()
    
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
    
            if args.kle_stop:
                if approx_kl > args.target_kl:
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        if args.kle_stop:
            writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        # PPG Storage:
        storage_slice = slice(args.num_steps*args.num_envs*(policy_update-1), args.num_steps*args.num_envs*policy_update)
        aux_obs[storage_slice] = b_obs.cpu().clone()
        aux_returns[storage_slice] = b_returns.cpu().clone()

    old_agent = Agent(envs).to(device)
    old_agent.load_state_dict(agent.state_dict())
    aux_inds = np.arange(args.aux_batch_size,)
    print("aux phase starts")
    for auxiliary_update in range(1, args.e_auxiliary+1):
        np.random.shuffle(aux_inds)
        for i, start in enumerate(range(0, args.aux_batch_size, args.aux_minibatch_size)):
            end = start + args.aux_minibatch_size
            aux_minibatch_ind = aux_inds[start:end]
            try:
                m_aux_obs = aux_obs[aux_minibatch_ind].to(device)
                m_aux_returns = aux_returns[aux_minibatch_ind].to(device)
                
                new_values = agent.get_value(m_aux_obs).view(-1)
                new_aux_values = agent.get_aux_value(m_aux_obs).view(-1)
                kl_loss = td.kl_divergence(old_agent.get_pi(m_aux_obs), agent.get_pi(m_aux_obs)).mean()
                
                real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                aux_value_loss = 0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                joint_loss = aux_value_loss + args.beta_clone * kl_loss
                
                optimizer.zero_grad()
                loss = (joint_loss+real_value_loss) / args.n_aux_grad_accum
                loss.backward()
                if (i+1) % args.n_aux_grad_accum == 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
            except RuntimeError:
                raise Exception ("if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory")
            
            del m_aux_obs, m_aux_returns
    writer.add_scalar("losses/aux/kl_loss", kl_loss.mean().item(), global_step)
    writer.add_scalar("losses/aux/aux_value_loss", aux_value_loss.item(), global_step)
    writer.add_scalar("losses/aux/real_value_loss", real_value_loss.item(), global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)

envs.close()
writer.close()
