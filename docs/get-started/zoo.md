# Model Zoo

CleanRL now has 🧪 experimental support for saving and loading models from 🤗 HuggingFace's [Model Hub](https://huggingface.co/models). We are rolling out this feature in phases, and currently only support saving and loading models from the following algorithm varaints:


| Algorithm      | Variants Implemented |
| ----------- | ----------- |
| ✅ [Deep Q-Learning (DQN)](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) | :material-github: [`dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqnpy) |
| | :material-github: [`dqn_atari.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_ataripy) |
| | :material-github: [`dqn_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_jaxpy) |
<!-- | | :material-github: [`dqn_atari_jax.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari_jax.py), :material-file-document: [docs](/rl-algorithms/dqn/#dqn_atari_jaxpy) | -->


## Load models from the Model Hub

We have a simple utility `enjoy.py` to load models from the hub and run them in an environment. We currently support the following commands:

```bash
poetry run python enjoy.py --exp-name dqn --env CartPole-v1
poetry run python enjoy.py --exp-name dqn_atari --env BreakoutNoFrameskip-v4
poetry run python enjoy.py --exp-name dqn_jax --env CartPole-v1
```

To see a list of supported models, please visit 🤗 https://huggingface.co/cleanrl.


## Save model to Model Hub

In the supported algorithm variant, you can run the script with the `--save-model` flag, which saves a model to the `runs` folder, and the `--upload-model` flag, which upload the model to huggingface under your default entity (username). Optionally, you may override the default entity with `--hf-entity` flag.

```bash
poetry run python cleanrl/dqn_jax.py --env-id CartPole-v1 --save-model --upload-model # --hf-entity cleanrl
poetry run python cleanrl/dqn_atari_jax.py --env-id SeaquestNoFrameskip-v4  --save-model --upload-model # --hf-entity cleanrl
```
