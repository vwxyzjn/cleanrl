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

### Design choices

CleanRL focues on providing single file implementatons of RL algorithms. It's salient characteristics would be as follows:

- The structure neural networks that model the various components of the RL algorithms (policy, value, ...) is usually hardcoded.

- Futhermore, agent rollout and training phases are essentially intertwined, following the convention of the reference implementations.

Due to the relatively higher complexity of Dreamer compared to *model-free* RL algorithms, this implementaton breaks some of the convention adopted so far in CleanRL's implementation:

- Hidden sizes of the neural networks can be parameterized from the command line to allow easier tuning and experimentation on different tasks.

- Uses a custom Replay Buffer that allows **Truncated BackPropagation Through Time (TBPTT)**. Collecton of the agent's rollout into this buffer is done implicitly with the `SampleCollectionWrapper` applied to the training environments. This is required to store the agent's trajectoroies in the specific forward that Dreamer training and TBPTT require. 

- Dreamer comes in two versions: Dreamer-v1 and Dreamer-v2. Dreamer-v2 uses **discrete stochastic latents** in the RSSM, as well as an implementation technicque referred to as **KL Balancing**. Instead of hard coding a specific version, this implementation allows toggling of these two features, while maintaining a relatively simple implementation. (@Costa: would it be better to have dreamerv1_XXX and dreamerv2_XXX instead ?)

- As mentioned earlier, the Dreamer agent can essentially be decomposed into the WM and the AC components. Those components are essentially orthogonal, at least during the training phase. This implementatoin encapsulates the WM and AC components into the `Dreamer` agent class. This class provides abstraction for sample action: `sample_action()` and for training `_train` to simplify the overview of the rollout training / process in `__main__()`. Each of `WorldModel` and `ActorCritic` class encapsulates the logic for its training. This was done with the hope of hastening the understand of what happens under the hood of each components independently, instead of hardcoding the interplay of both components during rollout and training.

- More walltime related training stats logging: since Dreamer agents can take a while to train, this implemetnaton has additional lines to provide more insight about the speed of the training process. On top of tracking the (agent) **steps per seconds (SPS)**, it also tracks:
    - **frames per seconds (FPS)**: in tasks such as Atari's Breakout, one action of the agent is usually repeated `env-action-repeat = 4` by default. Unlike CleanRL, the original paper report performance respectively to the number of frames sampled. The **FPS** metric, as well as the `global_frame` metric is tracked for compatibility of the result plotting.
    - **updates per seconds (UPS)**: backpropagating through time during Dreamer's training is the most time consuming process. This metric tracks how many update steps (i.e. calls to the `.backward()` method and `optimizer.step()`) are performed per second for the whole agent.
    - **Estimated Time of Arrival (ETA)**: it provides the estimated training time left for the experiment, based on the **UPS**. This metric is purely informative, and is not logged to Tensorboard / WandB. Instead, it is periodically printed to the console (when tracking with WandB, it can be seen in the "Logs" section of the run).
    - **PRGS**: Simiarly to **ETA**, this is used when tracking the epxeriment with WandB to mimic a *progress bar* that informs on the progression of the experiment.