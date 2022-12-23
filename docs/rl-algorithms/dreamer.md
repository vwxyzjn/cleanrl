# Dreamer

## Overivew

The family of methods referred to as _Dreamer agents_ is a set of of _model-based reinforcement learning methods_ that have seen a resurgence in the since 2019.

More rigorously, a _Dreamer_ agent is an hybrid of _model-based_ and _model-free_ RL algorithms.
Namely, it consists of a 1) **World Model** (WM) [TODO: Refs] component that learns the dynamics of a given task, and 2) an **Actor-Critic** (AC) [TODO: Refs] component that focus on the decision-making itself.

A *model-free* RL agent essentially maps from a given state to the action that is considered optimal, in a more or less **reflexive** manner, wihtout for example trying to account to future consequences of the selected actions.
On the other hand, conventional *model-based* RL method leverages a fully know, or learned WM to generate full plans (sequences of actions) and execute said plan into the real environment.

A *Dreamer* agent is thus and hybrid of *model-free* and *model-based* RL methods, in the sense that it can leverage the model to plan into the future, generate synthetic samples for training of the AC component, while at the same time generating actions and executing them in the environment following a closed loop fashion, based on a set of internal beliefs about the world that are sync with said environment [TODO: make this sentence easier to read].

As a *model-based* RL agent, Dreamer boast a better sample efficiency compared to *model-free* methods, and sometimes, even better performance.
This is oweing to the ability to leverage the learned model to generate synthetic samples that can be used to improve the decision-making component (policy) without requiring more environment samples.
This is an even more relevant property for task or environment where sampling can incur a high cost in time, but also lead to physical damage (like in autonomous driving).
Tangentially, learning models of the world model futher bridge the gap between RL agents and cognitive science (i.e. the ability of humnas to also model the world), and opens the door to more controlled studies of *learned representation*, *interpretability*, ... [**TODO**: flesh out this idea].
However, there is no free lunch, as this comes with a considerable trade-off in wall time.


### Original papers:
The inception of the *Dreamer agent* can be retraced as follows:

- Learning ...: introduces the *Recurrent State Space Model* (RSSM), a powerful model that can be used to plan directly from pixels

- Dreamer to Control: builds on top of the RSSM by introducing decision-making in the form of an AC compoent at the lower-dimensional, latent state space of the models, instead of using high-dimensional observatons such as pixel-based ones. The model is also designed to allow learning of policies by directly backpropagating through the WM itself.

- Mastering Atari with Discret World Models [TODO: Ref]: discretization of the latent states, which is more suited for some type of tasks, such as Atari games. Also introduces a tweak for learnign better dynamics overall (KL Balancing).

While the Dreamer agents implemented in this post are limited to up to the last reference, there is an additional, hierarchical variant of Dreamer referred to as *Director* that has also been proposed [TODO: Refs].

### Reference resources:
- Hafner's TF2.0 Implementation of Dreamer v2
- Hafner's TF1.XX Implementation of Dreamer v1, deprecated in favor of the aforementioned version.
- jsikyoon's Pytorch Implementation following Hafner's TF2.0 implementation.
- julius's pydreamer (note that they have a special way of doing the value update: GAE instead of Lambda returns)

Other potentially useful references that were not explored in this work
- Dreamer's Jax implementation by ???
- Kai's implementation

The original TF2.0 implementation, and its Pytorch equivalent are well organized, optimized, and written with modularity and re-use of various classes in mind (especially TF2.0 variant seems to have been written to allow a lot of ablations: detach gradient of each predictor heads, etc...).
However, this can be quite daunting for someone new to model-based RL methods.

This single file implementaton proposes a simpler approaches that trades off modularity, compatness of the code, and some amout of flexibility in favor of simplicity and understandability.

## `dreamer_atari.py`

## Design choices

CleanRL focues on providing single file implementatons of RL algorithms. It's salient characteristics would be as follows:

- The structure neural networks that model the various components of the RL algorithms (policy, value, ...) is usually hardcoded.

- Futhermore, agent rollout and training phases are essentially intertwined, following the convention of the reference implementations.

Due to the relatively higher complexity of Dreamer compared to *model-free* RL algorithms, this implementaton breaks some of the convention adopted so far in CleanRL's implementation:

- Hidden sizes of the neural networks can be parameterized from the command line to allow easier tuning and experimentation on different tasks.

- Uses a custom Replay Buffer that allows **Truncated BackPropagation Through Time (TBPTT)**. Collecton of the agent's rollout into this buffer is done implicitly with the `SampleCollectionWrapper` applied to the training environments. This is required to store the agent's trajectoroies in the specific forward that Dreamer training and TBPTT require. 

- Dreamer comes in two versions: Dreamer-v1 and Dreamer-v2. Dreamer-v2 uses **discrete stochastic latents** in the RSSM, as well as an implementation technicque referred to as **KL Balancing**. Instead of hard coding a specific version, this implementation allows toggling of these two features, while maintaining a relatively simple implementation. (@Costa: would it be better to have dreamerv1_XXX and dreamerv2_XXX instead ?)

- As mentioned earlier, the Dreamer agent can essentially be decomposed into the WM and the AC components. Those components are essentially orthogonal, at least during the training phase. This implementatoin encapsulates the WM and AC components into the `Dreamer` agent class. This class provides abstraction for sample action: `sample_action()` and for training `_train` to simplify the overview of the rollout training / process in `__main__()`. Each of `WorldModel` and `ActorCritic` class encapsulates the logic for its training. This was done with the hope of hastening the understand of what happens under the hood of each components independently, instead of hardcoding the interplay of both components during rollout and training.

## Underlying theory

### World Model (WM)
The main function of the WM component is to ...:

- learn low dimensional state representation from pixel, or more generally, observation data
- learn the dynamics of the task, which can then be used to
- generate synthetic trajectories to train the decision-making component

The WM is formed of the following components:

- **Encoder** $f_{\text{Encoder}}(o_t): \mathbb{O} \rightarrow \mathbb{S}$: maps from high-dimensional, pixel-based observations $o_t$ to a lower dimensional (feature) vector $x_t$
- **Decoder**: $f_{\text{Decoder}}(s_t): \mathbb{S} \rightarrow \mathbb{O}$: maps from low-dimensional state vector $s_t$ to an observation $\hat{o}_t$.
- **Reward predictor**: $\mathbb{S} \rightarrow \mathbb{R}$: maps from low-dimensional state vector $s_t$ to a real, scalar *reward* value.
- **Discount predictor**: $\mathbb{S} \rightarrow [0,1]$: maps from low-dimensional state vector $s_t$ to a scalar value in range $[0,1]$. This directly predicts the $\gamma$ discounting factor used for the agent's training

Following the RSSM [TODO: Refs.] structure, the low-dimensional state representation $s_t$ is formed by concatenation of a *deterministic component* hereafter denoted as $h_t$, and a *stochastic component* denoted as $y_t$.
The state transition function [...] deterministic component update.
This is done using Recurrent Neural Networks.
- **Deterministci state component update**: $f_{\text{RNN}(h_{t-1}, y_{t-1}, a_{t-1})}$.

The stochastic component $y_t$ of the state vector is approximated using either a *Normal* distribution in case of *continuous latents*, or a *OneHot Categorical* distribution in case of *discrete latents*.
- **Prior distribution over y_t**: $p(y_t \vert h_t)$: learns to predict the current stochaastic component of $s_t$ based on the history of the episode so far (which is encoded by h_t).
- **Posterior distribution over y_t**: $q(y_t \vert x_t, h_t)$: performs inference on the current $y_t$ based on the *observed feature vector* $x_t$, and the history so far encoded in $h_t$.

#### Training of the WM

- Assumes a sequence of observed data:
    - list of pixel-based observations: $\{o_t\}_{t=0}^{T-1}$
    - list of actions: $\{a_t\}_{t=0}^{T-1}$, and
    - list of episode termination signla: $\{d_t\}_{t=0}^{T-1}$
    For convenience, here is a diagram of how said data is structured when passed to the `_train()` method of the WM:
    **[TODO: Intutive diagram of the sequential observation data, with the action shifted one step right**
- The following diagram illustrates the overall training process
    **[TODO: Overall diagram of the training process**

### Actor-Critic (AC)

- based on the low-dimensional state space learned by the WM
- estimate the *value* of arbitrary states, which is then used to
- train a *policy* that produces actions maximizing the expected return.

It's component are as follows:
- **actor** network: policy $ \pi(a_t \vert s_t): \mathbb{S} \rightarrow \mathbb{A}$ that maps from low-dimensiona state vector $s_t$ to an action.
- **value** network: $V(s_t): \mathbb{S} \rightarrow \mathbb{R}$

### Explanation of the logged metrics

On top of the metrics conventional logged in CleanRL implementation, this implementation of Dreamer agents add other relevant metrics to the *model-based* nature of the algorithm:

- `charts/SPS`: is the number of **steps per second**, i.e. the number of `env.step()` calls made by the agent during rollout per second.

- `charts/FPS`: is the number of **frames per seconds (FPS)** sampled by the agent. This corresponds to `SPS * args.env_action_repeats`. Unlike CleanRL, the research papers report performance relatively to the number of **frames** sampled, instead of **environment steps**.
This metric is tracked for compatibilty of the result plotting, namely by using `global_frame` on the plots' horizontal axis.

- `charts/n_updates`: the number of updates performed on the overall Dreamer agent, i.e. the number of calls to `.backward()` and `optimizer.step()` on the WM and AC components.

- `charts/UPS`: the average number of model updates per second.

- `charts/PRGS`: tracks to progresion rate of the experiment, namely to be used with the Bar chart of WandB to mimic a progress bar.

Asside of the Tensorboard / WandB metrics, ... [TODO]

More walltime related training stats logging: since Dreamer agents can take a while to train, this implementation has additional lines to provide more insight about the speed of the training process.
Therefore, the running script will periodically printout a human-readable **Estimated Time of Arrival (ETA)**.
As the name suggests, it informs the user on how much time is lfet for the trainign to complete, and is computed based on the `charts/UPS` metric.
