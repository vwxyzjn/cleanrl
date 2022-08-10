# CleanRL - Overview


[<img src="https://img.shields.io/badge/license-MIT-blue">](https://github.com/vwxyzjn/cleanrl)
[![tests](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml/badge.svg)](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml)
[![docs](https://img.shields.io/github/deployments/vwxyzjn/cleanrl/Production?label=docs&logo=vercel)](https://docs.cleanrl.dev/)
[<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/D6RCjA6sVT)
[<img src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">](https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<!-- ## Overview -->

CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:

* 📜 Single-file implementation
   * *Every detail about an algorithm variant is put into a single standalone file.* 
   * For example, our `ppo_atari.py` only has 340 lines of code but contains all implementation details on how PPO works with Atari games, **so it is a great reference implementation to read for folks who do not wish to read an entire modular library**.
* 📊 Benchmarked Implementation (7+ algorithms and 34+ games at https://benchmark.cleanrl.dev)
* 📈 Tensorboard Logging
* 🪛 Local Reproducibility via Seeding
* 🎮 Videos of Gameplay Capturing
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 💸 Cloud Integration with docker and AWS 

You can read more about CleanRL in our [technical paper](https://arxiv.org/abs/2111.08819) and [documentation](https://docs.cleanrl.dev/).

Good luck have fun :rocket:

⚠️ **NOTE**: CleanRL is *not* a modular library and therefore it is not meant to be imported. At the cost of duplicate code, we make all implementation details of a DRL algorithm variant easy to understand, so CleanRL comes with its own pros and cons. You should consider using CleanRL if you want to 1) understand all implementation details of an algorithm's varaint or 2) prototype advanced features that other modular DRL libraries do not support (CleanRL has minimal lines of code so it gives you great debugging experience and you don't have do a lot of subclassing like sometimes in modular DRL libraries).

## Citing CleanRL

If you use CleanRL in your work, please cite our technical [paper](https://arxiv.org/abs/2111.08819):

```bibtex
@article{huang2021cleanrl,
    title={CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms}, 
    author={Shengyi Huang and Rousslan Fernand Julien Dossa and Chang Ye and Jeff Braga},
    year={2021},
    eprint={2111.08819},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
