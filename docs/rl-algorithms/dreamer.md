# Dreamer

## Overivew

The family of methods referred to as _Dreamer agents_ is a set of of _model-based reinforcement learning methods_ that have seen a resurgence in the since 2019.

More rigorously, a _Dreamer_ agent is an hybrid of _model-based_ and _model-free_ RL algorithms.
Namely, it consists of a 1) **World Model** (WM) [TODO: Refs] component that learns the dynamics of a given task, and 2) an **Actor-Critic** (AC) [TODO: Refs] component that focus on the decision-making itself.

A *model-free* RL agent essentially maps from a given state to the action that is considered optimal, in a more or less **reflexive** manner, wihtout for example trying to account to future consequences of the selected actions.
On the other hand, conventional *model-based* RL method leverages a fully know, or learned WM to generate full plans (sequences of actions) and execute said plan into the real environment.

A *Dreamer* agent is thus and hybrid of *model-free* and *model-based* RL methods, in the sense that it can leverage the model to plan into the future, generate synthetic samples for training of the AC component, while at the same time generating actions and executing them in the environment following a closed loop fashion, based on a set of internal beliefs about the world that are sync with said environment [TODO: make this sentence easier to read].

Original papers: the inception of the *Dreamer agent* can be retraced as follows:
- Learning ...: introduces the *Recurrent State Space Model* (RSSM), a powerful model that can be used to plan directly from pixels
- Dreamer to Control: builds on top of the RSSM by introducing decision-making in the form of an AC compoent at the lower-dimensional, latent state space of the models, instead of using high-dimensional observatons such as pixel-based ones. The model is also designed to allow learning of policies by directly backpropagating through the WM itself.
- Mastering Atari with Discret World Models [TODO: Ref]: discretization of the latent states, which is more suited for some type of tasks, such as Atari games. Also introduces a tweak for learnign better dynamics overall (KL Balancing).

While the Dreamer agents implemented in this post are limited to up to the last reference, there is an additional, hierarchical variant of Dreamer referred to as *Director* that has also been proposed [TODO: Refs].

Reference resources:
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

