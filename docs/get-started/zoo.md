# 🤗 Model Zoo

[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Huggingface-F8D521">](https://huggingface.co/cleanrl)
[![Open In Colab](https://github.com/vwxyzjn/cleanrl/raw/master/docs/get-started/colab-badge.svg)](https://colab.research.google.com/github/vwxyzjn/cleanrl/blob/master/docs/get-started/CleanRL_Huggingface_Integration_Demo.ipynb)

CleanRL now has 🧪 experimental support for saving and loading models from 🤗 HuggingFace's [Model Hub](https://huggingface.co/models). We are rolling out this feature in phases, and currently only support saving and loading models from the following algorithm variants:


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | :material-github: [`dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqnpy) |
| | :material-github: [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_ataripy) |
| | :material-github: [`dqn_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_jaxpy) |
| | :material-github: [`dqn_atari_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_atari_jaxpy) |
| ✅ [Categorical DQN (C51)](https://arxiv.org/pdf/1707.06887.pdf) | :material-github: [`c51.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py), :material-file-document: [docs](/rl-algorithms/c51/#c51py) |
| | :material-github: [`c51_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari.py), :material-file-document: [docs](/rl-algorithms/c51/#c51_ataripy) |
| | :material-github: [`c51_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_jax.py), :material-file-document: [docs](/rl-algorithms/c51/#c51_jaxpy) |
| | :material-github: [`c51_atari_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51_atari_jax.py), :material-file-document: [docs](/rl-algorithms/c51/#c51_atari_jaxpy) |
| ✅ [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf) | :material-github: [`ddpg_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action.py), :material-file-document: [docs](/rl-algorithms/ddpg/#ddpg_continuous_actionpy) |
| | :material-github: [`ddpg_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ddpg_continuous_action_jax.py),  :material-file-document: [docs](/rl-algorithms/ddpg/#ddpg_continuous_action_jaxpy)
| ✅ [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/pdf/1802.09477.pdf) | :material-github: [`td3_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_actionpy) |
|  | :material-github: [`td3_continuous_action_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py), :material-file-document: [docs](/rl-algorithms/td3/#td3_continuous_action_jaxpy) |


## Load models from the Model Hub

We have a simple utility `enjoy.py` to load models from the hub and run them in an environment. We currently support the following commands:

```bash
poetry install -E dqn
poetry run python -m cleanrl_utils.enjoy --exp-name dqn --env-id CartPole-v1
poetry install -E dqn_jax
poetry run python -m cleanrl_utils.enjoy --exp-name dqn_jax --env-id CartPole-v1

poetry install -E dqn_atari
poetry run python -m cleanrl_utils.enjoy --exp-name dqn_atari --env-id BreakoutNoFrameskip-v4
poetry install -E dqn_atari_jax
poetry run python -m cleanrl_utils.enjoy --exp-name dqn_atari_jax --env-id BreakoutNoFrameskip-v4
```

To see a list of supported models, please visit 🤗 [https://huggingface.co/cleanrl](https://huggingface.co/cleanrl).


???+ info "What happens under the hood?"
    
    The `cleanrl_utils.enjoy` is a simple wrapper to load the models from the hub and run them in an environment. A minimal version of the script can be found at [cleanrl_utils/evals/dqn_eval.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/evals/dqn_eval.py), which may give you a more fine-grained control and access to the model.
    
    <script src="https://emgithub.com/embed-v2.js?target=https%3A%2F%2Fgithub.com%2Fvwxyzjn%2Fcleanrl%2Fblob%2Fmaster%2Fcleanrl_utils%2Fevals%2Fdqn_eval.py&style=github&type=code&showBorder=on&showLineNumbers=on&showFileMeta=on&showFullPath=on&showCopy=on"></script>

## Save model to Model Hub

In the supported algorithm variants, you can run the script with the `--save-model` flag, which saves a model to the `runs` folder, and the `--upload-model` flag, which upload the model to huggingface under your default entity (username). Optionally, you may override the default entity with `--hf-entity` flag.

```bash
poetry run python cleanrl/dqn_jax.py --env-id CartPole-v1 --save-model --upload-model # --hf-entity cleanrl
poetry run python cleanrl/dqn_atari_jax.py --env-id SeaquestNoFrameskip-v4  --save-model --upload-model # --hf-entity cleanrl
```
