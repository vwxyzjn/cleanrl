# CleanRL - Get Started

[<img src="https://img.shields.io/badge/discord-cleanrl-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/D6RCjA6sVT)
[![Meeting Recordings : cleanrl](https://img.shields.io/badge/meeting%20recordings-cleanrl-green?logo=youtube&logoColor=ffffff&labelColor=FF0000&color=282828&style=flat?label=healthinesses)](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)
[<img src="https://github.com/vwxyzjn/cleanrl/workflows/build/badge.svg">](
https://github.com/vwxyzjn/cleanrl/actions)
[<img src="https://badge.fury.io/py/cleanrl.svg">](
https://pypi.org/project/cleanrl/)



CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:


* 📜 Single-file implementation
   * *Every detail about an algorithm is put into the algorithm's own file.* It is therefore easier to fully understand an algorithm and do research with.
* 📊 Benchmarked Implementation (7+ algorithms and 34+ games at https://benchmark.cleanrl.dev)
* 📈 Tensorboard Logging
* 🪛 Local Reproducibility via Seeding
* 🎮 Videos of Gameplay Capturing
* 🧫 Experiment Management with [Weights and Biases](https://wandb.ai/site)
* 💸 Cloud Integration with docker and AWS 

Good luck have fun 🚀

## Get started

Prerequisites:

* Python 3.8+
* [Poetry](https://python-poetry.org)

Simply run the following command for a quick start

```bash
git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
poetry install
poetry run python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --capture-video

# open another temrminal and enter `cd cleanrl/cleanrl`
tensorboard --logdir runs
# check out the videos of the agent's gameplay in the `videos` folder
```


<script id="asciicast-443622" src="https://asciinema.org/a/443622.js" async></script>




## Citing our project

Please consider using the following Bibtex entry:

```
@misc{cleanrl,
  author = {Shengyi Huang and Rousslan Dossa and Chang Ye},
  title = {CleanRL: High-quality Single-file Implementation of Deep Reinforcement Learning algorithms},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vwxyzjn/cleanrl/}},
}
```

## References

I have been heavily inspired by the many repos and blog posts. Below contains a incomplete list of them.

* http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
* https://github.com/seungeunrho/minimalRL
* https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On
* https://github.com/hill-a/stable-baselines

The following ones helped me a lot with the continuous action space handling:

* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py

