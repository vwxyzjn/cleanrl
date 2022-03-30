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
<a href="https://docs.cleanrl.dev/get-started/installation/">[Getting Started]</a>
- <a href="https://docs.cleanrl.dev/">[Docs]</a>
<!-- - <a href="https://docs.mosaicml.com/en/stable/method_cards/methods_overview.html">[Methods]</a>
- <a href="https://cleanrl.dev/team">[We're Hiring!]</a> -->
</p></h4>

<p align="center">
    <a href="https://img.shields.io/github/license/vwxyzjn/cleanrl">
        <img alt="Tests CI" src="https://github.com/vwxyzjn/cleanrl/actions/workflows/tests.yaml/badge.svg">
    </a>
    <a href="https://docs.cleanrl.dev/">
        <img alt="Docs" src="https://github.com/vwxyzjn/cleanrl/actions/workflows/docs.yaml/badge.svg">
    </a>
    <a href="https://discord.gg/D6RCjA6sVT">
        <img alt="Discord" src="https://img.shields.io/discord/767863440248143916?label=discord">
    </a>
    <a href="https://pypi.org/project/cleanrl/">
        <img alt="Documentation" src="https://badge.fury.io/py/cleanrl.svg">
    </a>
    <a href="https://www.youtube.com/channel/UCDdC6BIFRI0jvcwuhi3aI6w/videos">
        <img alt="Chat @ Slack" src="https://img.shields.io/youtube/channel/views/UCDdC6BIFRI0jvcwuhi3aI6w?style=social">
    </a>
    <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://pycqa.github.io/isort/">
        <img alt="Imports: isort" src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336">
    </a>
</p>

------



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
