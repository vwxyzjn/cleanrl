# CleanRL (Clean Implementation of RL Algorithms)

### This project is WIP currently at 0.2.1 release, expect breaking changes.

This repository focuses on a clean and minimal implementation of reinforcement learning algorithms that focuses on easy experimental research. The highlight features of this repo are:

* Most algorithms are self-contained in single files with a common dependency file [common.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/common.py) that handles different gym spaces.

* Easy logging of training processes using Tensorboard and Integration with wandb.com to log experiments on the cloud. Check out https://cleanrl.costa.sh.

* **Hackable** and being able to debug *directly* in Python’s interactive shell (Especially if you use the Spyder editor from Anaconda :) ).

* Convenient use of commandline arguments for hyper-parameters tuning.

* Benchmarked in many types of games. https://cleanrl.costa.sh

![wandb.png](wandb.png)

## Get started

To run experiments locally, give the following a try:

```bash
$ git clone https://github.com/vwxyzjn/cleanrl.git && cd cleanrl
$ pip install -e .
$ cd cleanrl
$ python a2c.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
# open another temrminal and enter `cd cleanrl/cleanrl`
$ tensorboard --logdir runs
```

![demo.gif](demo.gif)

To use wandb integration, sign up an account at https://wandb.com and copy the API key.
Then run

```bash
$ cd cleanrl
$ pip install wandb
$ wandb login ${WANBD_API_KEY}
$ python a2c.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --prod-mode True \
    --wandb-project-name cleanrltest 
# Then go to https://app.wandb.ai/${WANDB_USERNAME}/cleanrltest/
```

Checkout the demo sites at [https://app.wandb.ai/costa-huang/cleanrltest](https://app.wandb.ai/costa-huang/cleanrltest)

![demo2.gif](demo2.gif)

## Algorithms Implemented
- [x] Advantage Actor Critic (A2C)
    * [a2c.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c.py)
        * Discrete action space
    * [a2c_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c_continuous_action.py)
        * Continuous action space
- [x] Deep Q-Learning (DQN)
    * [dqn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py)
        * Discrete action space
    * [dqn_cnn.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_cnn.py)
        * Specifically for playing Atari games. It uses convolutional layers and other pre-processing techniques.
- [x] Soft Actor Critic (SAC)
    * [sac.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac.py)
        * Discrete action space
    * [sac_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py)
        * Continuous action space
- [x] Proximal Policy Gradient (PPO) 
    * [ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
        * Discrete action space
    * [ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
        * Continuous action space
    * [ppo2_continuous_actions.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo2_continuous_actions.py)
        * Also for continuous action space, but this script allows you to toggle following techniques
            * GAE (general advantage estimation)
            * Reward normalization and clipping
            * Observation normalization and clipping
            * KL divergence bounding
            * Learning rate annealing
            * Orthogonal layer initialization

## User's Guide for Researcher (Please read this if consider using CleanRL)

CleanRL focuses on early and mid stages of RL research, where one would try to understand ideas and do hacky experimentation with the algorithms. If your goal does not include messing with different parts of RL algorithms, perhaps library like [stable-baselines](https://github.com/hill-a/stable-baselines), [ray](https://github.com/ray-project/ray), or [catalyst](https://github.com/catalyst-team/catalyst) would be more suited for your use cases since they are built to be highly optimized, concurrent and fast.

CleanRL, however, is built to provide a simplified and streamlined approach to conduct RL experiment. Let's give an example. Say you are interested in implementing the [GAE (Generalized Advantage Estimation) technique](https://arxiv.org/abs/1506.02438) to see if it improves the A2C's performance on `CartPole-v0`. The workflow roughly looks like this:

1. Make a copy of `cleanrl/cleanrl/a2c.py` to `cleanrl/cleanrl/experiments/a2c_gae.py`
2. Implement the GAE technique. This should relatively simple because you don't have to navigate into dozens of files and find the some function named `compute_advantages()`
3. Run `python cleanrl/cleanrl/experiments/a2c_gae.py` in the terminal or using an interactive shell like [Spyder](https://www.spyder-ide.org/). The latter gives you the ability to stop the program at any time and execute arbitrary code; so you can program on the fly.
4. Open another terminal and type `tensorboard --logdir cleanrl/cleanrl/experiments/runs` and checkout the `episode_rewards`, `losses/policy_loss`, etc. If something appears not right, go to step 2 and continue.
5. If the technique works, you want to see if it works with other games such as `Taxi-v3` or different parameters as well. Execute 
    ```
    $ wandb login ${WANBD_API_KEY}
    $ for seed in {1..2}
        do
            (sleep 0.3 && nohup python a2c_gae.py \
            --seed $seed \
            --gym-id CartPole-v0 \
            --total-timesteps 30000 \
            --wandb-project-name myRLproject \
            --prod-mode True
            ) >& /dev/null &
        done
    $ for seed in {1..2}
        do
            (sleep 0.3 && nohup python a2c_gae.py \
            --seed $seed \
            --gym-id Taxi-v3 \   # different env
            --total-timesteps 30000 \
            --gamma 0.8 \ # a lower discount factor
            --wandb-project-name myRLproject \
            --prod-mode True
            ) >& /dev/null &
        done
    ```
    And then you can monitor the performances and keep good track of all the parameters used in your experiments
6. Continue this process

This pipline described above should give you an idea of how to use CleanRL for your research.

## Feature TODOs:

- [x] Add automatic benchmark 
    - Completed. See https://app.wandb.ai/costa-huang/cleanrl.benchmark/reports?view=costa-huang%2Fbenchmark
- [x] Support continuous action spaces
    - Preliminary support with [a2c_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/a2c_continuous_action.py)
- [x] Support using GPU
- [ ] Support using multiprocessing

## References

I have been heavily inspired by the many repos and blog posts. Below contains a incomplete list of them.

* http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
* https://github.com/seungeunrho/minimalRL
* https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On
* https://github.com/hill-a/stable-baselines

The following ones helped me a lot with the continuous action space handling:

* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/zhangchuheng123/Reinforcement-Implementation/blob/master/code/ppo.py
