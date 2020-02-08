import torch as ch
from torch.distributions.categorical import Categorical
import numpy as np
'''
Common functions/utilities implemented in PyTorch
Sorted into categories:
- General functions
- Actor-critic helpers
- Policy gradient (PPO/TRPO) helpers
- Normalization helpers
- Neural network helpers
- Initialization helpers
'''

########################
### GENERAL UTILITY FUNCTIONS:
# Parameters, unroll, cu_tensorize, cpu_tensorize, shape_equal_cmp,
# shape_equal, scat, determinant, safe_op_or_neg_one
########################

CKPTS_TABLE = 'checkpoints'


class Parameters():
    '''
    Parameters class, just a nice way of accessing a dictionary
    > ps = Parameters({"a": 1, "b": 3})
    > ps.A # returns 1
    > ps.B # returns 3
    '''
    def __init__(self, params):
        self.params = params
    
    def __getattr__(self, x):
        return self.params[x.lower()]

def unroll(*tensors):
    '''
    Utility function unrolling a list of tensors
    Inputs:
    - tensors; all arguments should be tensors (at least 2D))))
    Returns:
    - The same tensors but with the first two dimensions flattened
    '''
    rets = []
    for t in tensors:
        assert len(t.shape) >= 2
        new_shape = [t.shape[0]*t.shape[1]] + list(t.shape[2:])
        rets.append(t.contiguous().view(new_shape))
    return rets

def cu_tensorize(t):
    '''
    Utility function for turning arrays into cuda tensors
    Inputs:
    - t, list
    Returns:
    - Tensor version of t
    '''
    return ch.tensor(t).float().cuda()

def cpu_tensorize(t):
    '''
    Utility function for turning arrays into cpu tensors
    Inputs:
    - t, list
    Returns:
    - Tensor version of t
    '''
    return ch.tensor(t).float()

def gpu_mapper():
    return ch.device('cuda:0') if not cpu else ch.device('cpu')

def shape_equal_cmp(*args):
    '''
    Checks that the shapes of the passed arguments are equal
    Inputs:
    - All arguments should be tensors
    Returns:
    - True if all arguments have the same shape, else ValueError
    '''
    for i in range(len(args)-1):
        if args[i].shape != args[i+1].shape:
            s = "\n".join([str(x.shape) for x in args])
            raise ValueError("Expected equal shapes. Got:\n%s" % s)
    return True

def shape_equal(a, *args):
    '''
    Checks that a group of tensors has a required shape
    Inputs:
    - a, required shape for all the tensors
    - Rest of the arguments are tensors
    Returns:
    - True if all tensors are of shape a, otherwise ValueError
    '''
    for arg in args:
        if list(arg.shape) != list(a):
            if len(arg.shape) != len(a):
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
            for i in range(len(arg.shape)):
                if a[i] == -1 or a[i] == arg.shape[i]:
                    continue
                raise ValueError("Expected shape: %s, Got shape %s" \
                                    % (str(a), str(arg.shape)))
    return shape_equal_cmp(*args)

def scat(a, b, axis):
    '''
    Set-or-Cat (scat)
    Circumventing a PyTorch bug that auto-squeezes empty tensors.
    Inputs:
    a - A torch tensor, or None
    b - A torch tensor, can not be None
    axis - Axis to concat with
    Returns:
    - b if a is None, otherwise b concatted to a
    '''
    if a is None:
        return b
    return ch.cat((a, b), axis)

def determinant(mat):
    '''
    Returns the determinant of a diagonal matrix
    Inputs:
    - mat, a diagonal matrix
    Returns:
    - The determinant of mat, aka product of the diagonal
    '''
    return ch.exp(ch.log(mat).sum())

def safe_op_or_neg_one(maybe_empty, op):
    '''
    Performs an operation on a tensor which may be empty.
    Returns -1 if the tensor is empty, and returns the result
    of the op otherwise.
    Inputs:
    - maybe_empty, tensor which may be empty
    - op, an operation (tensor) -> (object) to perform
    Returns:
    - -1 if tensor is empty otherwise op(maybe_empty)
    '''
    if maybe_empty.nelement() == 0:
        return -1.
    else:
        return op(maybe_empty)

########################
### ACTOR-CRITIC HELPERS:
# discount_path, get_path_indices, select_prob_dists
########################

# Can be used to convert rewards into discounted returns:
# ret[i] = sum of t = i to T of gamma^(t-i) * rew[t]
def discount_path(path, h):
    '''
    Given a "path" of items x_1, x_2, ... x_n, return the discounted
    path, i.e. 
    X_1 = x_1 + h*x_2 + h^2 x_3 + h^3 x_4
    X_2 = x_2 + h*x_3 + h^2 x_4 + h^3 x_5
    etc.
    Can do (more efficiently?) w SciPy. Python here for readability
    Inputs:
    - path, list/tensor of floats
    - h, discount rate
    Outputs:
    - Discounted path, as above
    '''
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr*h + path[-1-i]
        rets.append(curr)
    rets =  ch.stack(list(reversed(rets)), 0)
    return rets

def get_path_indices(not_dones):
    """
    Returns list of tuples of the form:
        (agent index, time index start, time index end + 1)
    For each path seen in the not_dones array of shape (# agents, # time steps)
    E.g. if we have an not_dones of composition:
    tensor([[1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]], dtype=torch.uint8)
    Then we would return:
    [(0, 0, 3), (0, 3, 10), (1, 0, 3), (1, 3, 5), (1, 5, 9), (1, 9, 10)]
    """
    indices = []
    num_timesteps = not_dones.shape[1]
    for actor in range(not_dones.shape[0]):
        last_index = 0
        for i in range(num_timesteps):
            if not_dones[actor, i] == 0.:
                indices.append((actor, last_index, i + 1))
                last_index = i + 1
        if last_index != num_timesteps:
            indices.append((actor, last_index, num_timesteps))
    return indices

def select_prob_dists(pds, selected=None, detach=True):
    '''
    Given a tensor/tuple probability distributions, and 
    some indices, select a subset of the distributions 
    `pds`s according to the indices `selected`.
    Inputs:
    - pds: list of propo
    '''
    if type(pds) is tuple:
        if selected is not None:
            tup = (pds[0][selected], pds[1])
        else:
            tup = pds
        return tuple(x.detach() if detach else x for x in tup)
    out = pds[selected] if selected is not None else pds
    return out.detach() if detach else out
        

########################
### POLICY GRADIENT HELPERS:
# vjp, jvp, cg_solve, backtracking_line_search
########################

def vjp(f_x, theta, v, create=True):
    '''
    Vector-jacobian product
    Calculates v^TJ, or J^T v, using standard backprop
    Input:
    - f_x, function of which we want the Jacobian
    - theta, variable with respect to which we want Jacobian
    - v, vector that we want multiplied by the Jacobian
    Returns:
    - J^T @ v, without using n^2 space
    '''
    grad_list = ch.autograd.grad(f_x, theta, v, retain_graph=True, create_graph=create)
    return ch.nn.utils.parameters_to_vector(grad_list)

def jvp(f_x, theta, v):
    '''
    Jacobian-vector product
    Calculate the Jacobian-vector product, see
    https://j-towns.github.io/2017/06/12/A-new-trick.html for math
    Input:
    - f_x, function of which we want the Jacobian
    - theta, variable with respect to which we want Jacobian
    - v, vector that we want multiplied by the Jacobian
    Returns:
    - J @ v, without using n^2 space
    '''
    w = ch.ones_like(f_x, requires_grad=True)
    JTw = vjp(f_x, theta, w)
    return vjp(JTw, w, v)

def cg_solve(fvp_func, b, nsteps):
    '''
    Conjugate Gradients Algorithm
    Solves Hx = b, where H is the Fisher matrix and b is known
    Input:
    - fvp_func, a callable function returning Fisher-vector product
    - b, the RHS of the above
    - nsteps, the number of steps on CG to take
    Returns:
    - An approximate solution x of Hx = b
    '''
    # Initialize the solution, residual, direction vectors
    x = ch.zeros(b.size()) 
    r = b.clone()
    p = b.clone()
    new_rnorm = ch.dot(r,r)
    for _ in range(nsteps):
        rnorm = new_rnorm
        fvp = fvp_func(p)
        alpha = rnorm / ch.dot(p, fvp)
        x += alpha * p
        r -= alpha * fvp
        new_rnorm = ch.dot(r, r)
        ratio = new_rnorm / rnorm
        p = r + ratio * p
    return x

def backtracking_line_search(f, x, expected_improve_rate, 
                             num_tries=10, accept_ratio=.1):
    '''
    Backtracking Line Search
    Inputs:
    - f, function for improvement of the objective
    - x, biggest step to try (successively halved)
    - num_tries, number of times to try halving x before giving up
    - accept_ratio, how much of the expected improve rate we have to
    improve by
    '''
    # f gives improvement
    for i in range(num_tries):
        scaling = 2**(-i)
        scaled = x * scaling
        improve = f(scaled)
        expected_improve = expected_improve_rate * scaling
        if improve/expected_improve > accept_ratio and improve > 0: 
            print("We good! %f" % (scaling,))
            return scaled
    return 0.

########################
### NORMALIZATION HELPERS:
# RunningStat, ZFilter, StateWithTime
########################

class RunningStat(object):
    '''
    Keeps track of first and second moments (mean and variance)
    of a streaming time series.
     Taken from https://github.com/joschu/modular_rl
     Math in http://www.johndcook.com/blog/standard_deviation/
    '''
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape

class Identity:
    '''
    A convenience class which simply implements __call__
    as the identity function
    '''
    def __call__(self, x, *args, **kwargs):
        return x

    def reset(self):
        pass

class RewardFilter:
    """
    Incorrect reward normalization [copied from OAI code]
    update return
    divide reward by std(return) without subtracting and adding back mean
    """
    def __init__(self, prev_filter, shape, gamma, clip=None):
        assert shape is not None
        self.gamma = gamma
        self.prev_filter = prev_filter
        self.rs = RunningStat(shape)
        self.ret = np.zeros(shape)
        self.clip = clip

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.ret = self.ret * self.gamma + x
        self.rs.push(self.ret)
        x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def reset(self):
        self.ret = np.zeros_like(self.ret)
        self.prev_filter.reset()

class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """
    def __init__(self, prev_filter, shape, center=True, scale=True, clip=None):
        assert shape is not None
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)
        self.prev_filter = prev_filter

    def __call__(self, x, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.rs.push(x)
        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff/(self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    def reset(self):
        self.prev_filter.reset()

class StateWithTime:
    '''
    Keeps track of the time t in an environment, and 
    adds t/T as a dimension to the state, where T is the 
    time horizon, given at initialization.
    '''
    def __init__(self, prev_filter, horizon):
        self.counter = 0
        self.horizon = horizon
        self.prev_filter = prev_filter

    def __call__(self, x, reset=False, count=True, **kwargs):
        x = self.prev_filter(x, **kwargs)
        self.counter += 1 if count else 0
        self.counter = 0 if reset else self.counter
        return np.array(list(x) + [self.counter/self.horizon,])

    def reset(self):
        self.prev_filter.reset()

# TODO: redo this in a not fucked way (ie using python language features)
class Trajectories:
    def __init__(self, states=None, rewards=None, returns=None, not_dones=None,
                 actions=None, action_log_probs=None, advantages=None,
                 unrolled=False, values=None):

        self.states = states
        self.rewards = rewards
        self.returns = returns
        self.values = values
        self.not_dones = not_dones
        self.actions = actions
        self.action_log_probs = action_log_probs
        self.advantages = advantages
        self.unrolled = unrolled

        # this is disgusting and we should fix it
        if states is not None:
            num_saps = states.shape[0]
            assert states is None or states.shape[0] == num_saps
            assert rewards is None or rewards.shape[0] == num_saps
            assert returns is None or returns.shape[0] == num_saps
            assert values is None or values.shape[0] == num_saps
            assert not_dones is None or not_dones.shape[0] == num_saps
            assert actions is None or actions.shape[0] == num_saps
            assert action_log_probs is None or action_log_probs.shape[0] == num_saps
            assert advantages is None or advantages.shape[0] == num_saps

            self.size = num_saps
        
    def unroll(self):
        assert not self.unrolled
        return self.tensor_op(unroll, should_wrap=False)

    def tensor_op(self, lam, should_wrap=True):
        if should_wrap:
            def op(*args):
                return [lam(v) for v in args]
        else:
            op = lam

        tt = op(self.states, self.rewards, self.returns, self.not_dones)
        tt2 = op(self.actions, self.action_log_probs, self.advantages)
        values, = op(self.values)

        ts = Trajectories(states=tt[0], rewards=tt[1], returns=tt[2],
                          not_dones=tt[3], actions=tt2[0],
                          action_log_probs=tt2[1], advantages=tt2[2],
                          values=values, unrolled=True)

        return ts

########################
### NEURAL NETWORK HELPERS:
# orthogonal_init
########################

def orthogonal_init(tensor, gain=1):
    '''
    Fills the input `Tensor` using the orthogonal initialization scheme from OpenAI
    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> orthogonal_init(w)
    '''
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor
