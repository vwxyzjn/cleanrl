import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from cleanrl.common import preprocess_obs_space, preprocess_ac_space
import argparse
import numpy as np
import gym
import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

# Tensorflow support
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tf.disable_eager_execution()

# MODIFIED: Import buffer with random batch sampling support
from cleanrl.buffers import SimpleReplayBuffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAC with 2 Q functions, Online updates')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).strip(".py"),
                       help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v0",
                       help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=7e-4,
                       help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                       help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=1000,
                       help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=int( 1e6),
                       help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=bool, default=True,
                       help='whether to set `torch.backends.cudnn.deterministic=True`')
    parser.add_argument('--cuda', type=bool, default=True,
                       help='whether to use CUDA whenever possible')
    parser.add_argument('--prod-mode', type=bool, default=False,
                       help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                       help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help="the entity (team) of wandb's project")
    parser.add_argument('--notb', action='store_true',
       help='No Tensorboard logging')

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int( 1e5),
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='the discount factor gamma')
    parser.add_argument('--target-update-interval', type=int, default=1,
                       help="the timesteps it takes to update the target network")
    parser.add_argument('--batch-size', type=int, default=64,
                       help="the batch size of sample from the reply memory")
    parser.add_argument('--tau', type=float, default=0.005,
                       help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--alpha', type=float, default=0.2,
                       help="Entropy regularization coefficient.")
    parser.add_argument('--autotune', action='store_true',
        help='Enables autotuning of the alpha entropy coefficient')

    # Neural Network Parametrization
    parser.add_argument('--policy-hid-sizes', nargs='+', type=int, default=(120,84,))
    parser.add_argument('--value-hid-sizes', nargs='+', type=int, default=(120,84,))
    parser.add_argument('--q-hid-sizes', nargs='+', type=int, default=(120,84,))

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
input_shape, preprocess_obs_fn = preprocess_obs_space(env.observation_space, device)
output_shape = preprocess_ac_space(env.action_space)
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"

# ALGO LOGIC: initialize agent here:
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Placeholders
obs_ph = tf.placeholder( tf.float32, [None, input_shape], name="observations")
next_obs_ph = tf.placeholder( tf.float32, [None, input_shape], name="next_observations")
act_ph = tf.placeholder( tf.int32, [None], name="actions")
rew_ph = tf.placeholder( tf.float32, [None], name="rewards")
ter_ph = tf.placeholder( tf.float32, [None], name="terminals")
act_indices_ph = tf.placeholder( tf.int32, [None, 2], name="action_gather_indices")

# TODO: Add graph support
tf2.random.set_seed(args.seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session( config=config)
sess.as_default()

# Custom Categorical Policy
class Policy():
    def __init__(self, name="policy", sess = None):
        # Building the policy graph
        self.action_logits = obs_ph
        self.sess = sess

        with tf.variable_scope( name):
            for hsize in list( args.policy_hid_sizes):
                self.action_logits = tf.layers.dense( self.action_logits, hsize,
                    activation=tf.nn.relu)

            self.action_logits = tf.layers.dense( self.action_logits, output_shape)

        self.action_dist = tfp.distributions.Categorical( logits=self.action_logits,
            allow_nan_stats=False)

        self.actions = self.action_dist.sample()
        self.deterministic_actions = tf.argmax( self.action_logits, axis=-1) # TODO: Why axis=-1

        self.action_probs = tf.nn.softmax( self.action_logits, axis=-1) # TODO: Why axis -1 ?
        self.logps = tf.nn.log_softmax( self.action_logits, axis=-1) # TODO: Why again ?
        self.entropy_mean = tf.reduce_mean( self.action_dist.entropy())

        # Handling next observation related ops, Target values
        self.next_action_logits = next_obs_ph
        with tf.variable_scope( name, reuse=tf.AUTO_REUSE):
            for hsize in list( args.policy_hid_sizes):
                self.next_action_logits = tf.layers.dense( self.next_action_logits, hsize,
                    activation=tf.nn.relu)

            self.next_action_logits = tf.layers.dense( self.next_action_logits, output_shape)

        self.next_action_dist = tfp.distributions.Categorical( logits=self.next_action_logits,
            allow_nan_stats=False)

        self.next_action = self.next_action_dist.sample()

        self.next_action_probs = tf.nn.softmax( self.next_action_logits, axis=-1)
        self.next_logps = tf.nn.log_softmax( self.next_action_logits, axis=-1)

    def get_actions( self, observations, deterministic=False):

        action_op = self.actions
        if deterministic:
            action_op = self.deterministic_actions

        return self.sess.run( action_op, feed_dict={ obs_ph: observations})

class QValue():
    def __init__(self, input_ph, name="q", sess = None):
        self.action_values = input_ph
        self.sess = sess

        with tf.variable_scope( name):
            for hsize in list( args.q_hid_sizes):
                self.action_values = tf.layers.dense( self.action_values, hsize,
                    activation=tf.nn.relu)

            self.action_values = tf.layers.dense( self.action_values, output_shape,
                activation=None)

            # self.action_values = tf.squeeze( self.action_values)

        # 1 because action dim is always 1
        # gather_indices =

        self.state_action_values = tf.gather_nd( self.action_values, act_indices_ph)
        # self.state_action_values = tf.gather_nd( self.action_values, act_ph)

buffer = SimpleReplayBuffer( env.observation_space, env.action_space, args.buffer_size, args.batch_size)
buffer.set_seed( args.seed) # Seedable buffer for reproducibility

# Helper function
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

with tf.variable_scope( "main"):
    # Defining the agent's policy: Gaussian
    pg = Policy( sess=sess)

    # Defining the agent's policy: Gaussian
    qf1 = QValue( obs_ph, name="q1", sess=sess)
    qf2 = QValue( obs_ph, name="q2", sess=sess)

with tf.variable_scope( "target"):
    qf1_target = QValue( next_obs_ph, name="q1", sess=sess)
    qf2_target = QValue( next_obs_ph, name="q2", sess=sess)

# MODIFIED: Helper function to update target value function network
init_target_op = tf.group([ tf.assign( target_param, param)
    for target_param, param in zip( get_vars( "target/q"), get_vars( "main/q")) ])

update_target_op = tf.group([ tf.assign( target_param, target_param * (1-args.tau) + param * (args.tau))
    for target_param, param in zip( get_vars( "target/q"), get_vars( "main/q")) ])

# TODO: Add TF support for autoenttun
# MODIFIED: SAC Automatic Entropy Tuning support
if args.autotune:
    # This is only an Heuristic of the minimal entropy we should constraint to
    # TODO: Find a better heuristic
    target_entropy = tf.constant( - np.prod( env.action_space.shape), dtype=tf.float32,name="target_entropy")
    log_alpha = tf.get_variable( "log_alpha", initializer=[0.,])
    alpha = tf.exp( log_alpha)
else:
    alpha = args.alpha

# Predefining losses
# Q Values losses
min_qf_target_values = tf.minimum( qf1_target.action_values, qf2_target.action_values)
min_qf_target_values_2 = min_qf_target_values - alpha * pg.next_action_probs

v_next_approx = tf.reduce_sum( pg.next_action_probs * min_qf_target_values_2, 1)
# v_next_approx = tf.reduce_mean( pg.next_action_probs * min_qf_target_values_2, 1)

q_backup = rew_ph + ( 1. - ter_ph) * args.gamma * v_next_approx
q_backup = tf.stop_gradient( q_backup)

# TODO: Debug this with real values
qf1_loss_op = tf.reduce_mean( .5 * (tf.square( qf1.state_action_values - q_backup)))
qf2_loss_op = tf.reduce_mean( .5 * (tf.square( qf2.state_action_values - q_backup)))
q_loss_op = qf2_loss_op + qf1_loss_op

q_update_op = tf.train.AdamOptimizer( args.learning_rate).minimize( q_loss_op, var_list = get_vars( "main/q"))

# Policy loss
min_qf_values = tf.minimum( qf1.action_values, qf2.action_values)
policy_loss_op = alpha * pg.logps - min_qf_values
# policy_loss_op = tf.reduce_mean( pg.action_probs * policy_loss_op, 1)
policy_loss_op = tf.reduce_sum( pg.action_probs * policy_loss_op, 1)
policy_loss_op = tf.reduce_mean( policy_loss_op)

p_update_op = tf.train.AdamOptimizer( args.learning_rate).minimize( policy_loss_op, var_list = get_vars( "main/policy"))

if args.autotune:
    alpha_loss_op = pg.action_probs * (- alpha * (pg.logps + target_entropy))
    alpha_loss_op = tf.reduce_mean(tf.reduce_sum( alpha_loss_op, 1))

    a_update_op = tf.train.AdamOptimizer( args.learning_rate).minimize( alpha_loss_op, var_list = [log_alpha])

# Init all vars in the graphs
sess.run( tf.global_variables_initializer())
# Sync weights of the QValues
sess.run( init_target_op)


# Helper function to evaluate agent determinisitically
def test_agent( env, policy, eval_episodes=1):
    returns = []
    lengths = []

    for eval_ep in range( eval_episodes):
        ret = 0.
        done = False
        t = 0

        obs = np.array( env.reset())

        while not done:
            with torch.no_grad():
                action = pg.get_actions([obs]).tolist()[0]

            obs, rew, done, _ = env.step( action)
            obs = np.array( obs)
            ret += rew
            t += 1
        # TODO: Break if max episode length is breached

        returns.append( ret)
        lengths.append( t)

    return returns, lengths

# TRY NOT TO MODIFY: start the game
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

# MODIFIED: When testing, skip Tensorboard log creation
if not args.notb:
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, tensorboard=True, config=vars(args), name=experiment_name)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    wandb.save(os.path.abspath(__file__))

global_step = 0
while global_step < args.total_timesteps:
    next_obs = np.array(env.reset())

    # MODIFIED: Keeping track of train episode returns and lengths
    train_episode_return = 0.
    train_episode_length = 0

    done = False

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(args.episode_length):
        global_step += 1
        obs = next_obs.copy()

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            action = pg.get_actions([obs]).tolist()[0]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rew, done, _ = env.step(action)
        next_obs = np.array(next_obs)

        buffer.add_transition(obs, action, rew, done, next_obs)
        # TODO: Add custom buffer

        # Keeping track of train episode returns
        train_episode_return += rew
        train_episode_length += 1

        # ALGO LOGIC: training.
        if buffer.is_ready_for_sample:
            # TODO: Cast action batch to ints ?
            observation_batch, action_batch, reward_batch, \
                terminals_batch, next_observation_batch = buffer.sample(args.batch_size)
            # Trick to use gathed_nd, before we figure out how to use tf.gather itself
            action_batch_gindices = [ [batch_idx, int( a)] for batch_idx, a in enumerate( action_batch)]

            feed_dict ={ obs_ph: observation_batch, act_ph: action_batch,
                rew_ph: reward_batch, ter_ph: terminals_batch,
                next_obs_ph: next_observation_batch,
                act_indices_ph: action_batch_gindices
            }

            action_probs, logps = sess.run( [pg.action_probs, pg.logps],
                feed_dict=feed_dict)
            qf1_values = sess.run( qf1.action_values, feed_dict=feed_dict)
            min_qf_values_res = sess.run( min_qf_values, feed_dict=feed_dict)
            policy_loss_res = sess.run( policy_loss_op, feed_dict=feed_dict)

            # All in one updates
            all_ops = [qf1_loss_op, qf2_loss_op, policy_loss_op, pg.entropy_mean, q_update_op, p_update_op]
            qf1_loss, qf2_loss, policy_loss, entropy_mean, _, _ = sess.run( all_ops, feed_dict=feed_dict)

            if args.autotune:
                alpha_loss, _ = sess.run( [alpha_loss_op, a_update_op], feed_dict=feed_dict)

            if global_step > 0 and global_step % args.target_update_interval == 0:
                sess.run( update_target_op)

            # Some verbosity and logging
            # Evaulating in deterministic mode after one episode
            if global_step % args.episode_length == 0:
                eval_returns, eval_ep_lengths = test_agent( env, pg, 5)
                eval_return_mean = np.mean( eval_returns)
                eval_ep_length_mean = np.mean( eval_ep_lengths)

                # Log to TBoard
                if not args.notb:
                    writer.add_scalar("eval/episode_return", eval_return_mean, global_step)
                    writer.add_scalar("eval/episode_length", eval_ep_length_mean, global_step)

            if not args.notb:
                writer.add_scalar("train/q1_loss", qf1_loss, global_step)
                writer.add_scalar("train/q2_loss", qf2_loss, global_step)
                writer.add_scalar("train/policy_loss", policy_loss, global_step)
                writer.add_scalar("train/entropy", entropy_mean, global_step)

                if args.autotune:
                    writer.add_scalar("train/alpha_entropy_coef", sess.run( alpha), global_step)

            if global_step > 0 and global_step % 100 == 0:
                print( "Step %d: Poloss: %.6f -- Q1Loss: %.6f -- Q2Loss: %.6f"
                    % ( global_step, policy_loss, qf1_loss, qf2_loss))

        if done:
            # MODIFIED: Logging the trainin episode return and length, then resetting their holders
            if not args.notb:
                writer.add_scalar("eval/train_episode_return", train_episode_return, global_step)
                writer.add_scalar("eval/train_episode_length", train_episode_length, global_step)

            train_episode_return = 0.
            train_episode_length = 0

            break;

writer.close()
