# Model Zoo

:octicons-beaker-24: Experimental

CleanRL now has experimental support for saving and loading models from HuggingFace's [Model Hub](https://huggingface.co/models). We are rolling out this feature in phases, and currently only support saving and loading models from the following algorithms:


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | :material-github: [`dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqnpy) |
| | :material-github: [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_ataripy) |
| | :material-github: [`dqn_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_jaxpy) |
<!-- | | :material-github: [`dqn_atari_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_atari_jaxpy) | -->


## Load models from the Model Hub

We have a simple utility `enjoy.py` to load models from the hub and run them in an environment. We currently support the following commands:

```bash
python enjoy.py --exp-name dqn --env CartPole-v1
python enjoy.py --exp-name dqn_jax --env CartPole-v1
```