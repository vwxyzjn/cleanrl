# CleanRL

<br />
<p align="center">
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
    <a href="https://github.com/mosaicml/composer#gh-dark-mode-only" class="only-dark">
      <img src="static/logo.svg" width="35%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>
<p align="center">
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_BEGIN -->
    <a href="https://github.com/mosaicml/composer#gh-dark-mode-only" class="only-dark">
      <img src="static/logo2.svg" width="35%"/>
    </a>
    <!-- SETUPTOOLS_LONG_DESCRIPTION_HIDE_END -->
</p>

<h2><p align="center">Clean Implementation of RL Algorithms</p></h2>
<!-- <h3><p align="center">Train Faster, Reduce Cost, Get Better Models</p></h3> -->

<h4><p align='center'>
<a href="https://cleanrl.dev">[Website]</a>
- <a href="https://docs.cleanrl.dev/get-started/installation/">[Getting Started]</a>
- <a href="https://docs.cleanrl.dev/">[Docs]</a>
<!-- - <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://cleanrl.dev/team">[We're Hiring!]</a> -->
</p></h4>

<p align="center">
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Version" src="https://img.shields.io/pypi/pyversions/mosaicml">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Package Version" src="https://img.shields.io/pypi/v/mosaicml">
    </a>
    <a href="https://pypi.org/project/mosaicml/">
        <img alt="PyPi Downloads" src="https://img.shields.io/pypi/dm/mosaicml">
    </a>
    <a href="https://docs.mosaicml.com/en/stable/">
        <img alt="Documentation" src="https://readthedocs.org/projects/composer/badge/?version=stable">
    </a>
    <a href="https://join.slack.com/t/mosaicml-community/shared_invite/zt-w0tiddn9-WGTlRpfjcO9J5jyrMub1dg">
        <img alt="Chat @ Slack" src="https://img.shields.io/badge/slack-chat-2eb67d.svg?logo=slack">
    </a>
    <a href="https://github.com/mosaicml/composer/blob/dev/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=slack">
    </a>
</p>
<br />



<img src="
https://img.shields.io/github/license/vwxyzjn/cleanrl">
[![tests](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml/badge.svg)](https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml)
[![ci](https://github.com/vwxyzjn/cleanrl/actions/workflows/docs.yaml/badge.svg)](https://github.com/vwxyzjn/cleanrl/actions/workflows/docs.yaml)
[<img src="https://img.shields.io/discord/767863440248143916?label=discord">](https://discord.gg/D6RCjA6sVT)
[<img src="https://badge.fury.io/py/cleanrl.svg">](
https://pypi.org/project/cleanrl/)
[<img src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">](https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## Overview

CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:


* Single-file Implementation
    * **Every detail about an algorithm is put into the algorithm's own file.** Therefore, it's easier for you to fully understand an algorithm and do research with it.
* Benchmarked Implementation on 7+ algorithms and 34+ games 
* Tensorboard Logging
* Local Reproducibility via Seeding
* Videos of Gameplay Capturing
* Experiment Management with [Weights and Biases](https://wandb.ai/site)
* Cloud Integration with Docker and AWS 

You can read more about CleanRL in our [technical paper](https://arxiv.org/abs/2111.08819) and [documentation](https://docs.cleanrl.dev/).

Good luck have fun 🚀

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
