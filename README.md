# CleanRL (Clean Implementation of RL Algorithms)

[<img src="https://img.shields.io/badge/slack-@cleanrl-yellow.svg?logo=slack">](https://join.slack.com/t/cleanrl/shared_invite/zt-cj64t5eq-xKZ6sD0KPGFKu1QicHEvVg)
[![Mailing List : cleanrl](https://img.shields.io/badge/mailing%20list-cleanrl-orange.svg)](https://groups.google.com/forum/#!forum/rlimplementation/join)
[![Meeting Recordings : cleanrl](https://img.shields.io/badge/meeting%20recordings-cleanrl-orange.svg)](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)

CleanRL dedicates to be the most user-friendly Reinforcement Learning library. The implementation is clean, simple, and *self-contained*; you don't have to look through dozens of files to understand what is going on. Just read, print out a few things and you can easily customize.

At the same time, CleanRL tries to supply many research-friendly features such as cloud experiment management, support for continuous and discrete observation and action spaces, video recording of the game play, etc. These features will be very helpful for doing research, especially the video recording feature that *allows you to visually inspect the agents' behavior at various stages of the training*.

Good luck have fun :rocket:


### This project is WIP currently at 0.2.1 release, expect breaking changes.

The highlight features of this repo are:

* Most algorithms are self-contained in single files with a common dependency file [common.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/common.py) that handles different gym spaces.
* Easy logging of training processes using Tensorboard and Integration with wandb.com to log experiments on the cloud. Check out https://cleanrl.costa.sh.
* **Hackable** and being able to debug *directly* in Python’s interactive shell (Especially if you use the Spyder editor from Anaconda :) ).
* Simple use of command line arguments for hyper-parameters tuning; no need for arcane configuration files.

## Benchmarked Implementation

Our implementation is benchmarked to ensure quality. We log all of our benchmarked experiments using wandb so that you can check the hyper-parameters, videos of the agents playing the game, and the exact commands to reproduce it. See https://cleanrl.costa.sh.

<img src="wandb.png">


The current dashboard of wandb does not allow us to show the agents performance in all the games at the same panel, so you have to click each panel in https://cleanrl.costa.sh to check the benchmarked performance, which can be inconvenient sometimes. So we additionally post the benchmarked performance for each game using seaborn as follows (the result is created by using [`benchmark/plot_benchmark.py`](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark/plot_benchmark.py)

<img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/legend.svg">

<table>
    <tr>
        <th align="center">Benchmarked Learning Curves</th>
        <th align="center">Atari</th>
    </tr>
    <tr>
        <td align="center" colspan="2">Metrics, logs, and recorded videos are at https://app.wandb.ai/cleanrl/cleanrl.benchmark/reports/Atari--VmlldzoxMTExNTI</td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/QbertNoFrameskip-v4.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/BeamRiderNoFrameskip-v4.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/SpaceInvadersNoFrameskip-v4.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/PongNoFrameskip-v4.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/BreakoutNoFrameskip-v4.svg"></td>
        <td></td>
    </tr>
</table>


<table>
    <tr>
        <th align="center">Benchmarked Learning Curves</th>
        <th align="center">Mujoco</th>
    </tr>
    <tr>
        <td align="center" colspan="2">Metrics, logs, and recorded videos are at https://app.wandb.ai/cleanrl/cleanrl.benchmark/reports/Mujoco--VmlldzoxODE0NjE</td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Reacher-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/InvertedPendulum-v2.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Hopper-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Pusher-v2.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Striker-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Thrower-v2.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Ant-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/HalfCheetah-v2.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Walker2d-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Swimmer-v2.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Humanoid-v2.svg"></td>
        <td></td>
    </tr>
</table>

<table>
    <tr>
        <th align="center">Benchmarked Learning Curves</th>
        <th align="center">PyBullet and Other Continuous Action Tasks</th>
    </tr>
    <tr>
        <td align="center" colspan="2">Metrics, logs, and recorded videos are at https://app.wandb.ai/cleanrl/cleanrl.benchmark/reports/PyBullet-and-Other-Continuous-Action-Tasks--VmlldzoxODE0NzY</td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/InvertedDoublePendulumBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/MinitaurBulletDuckEnv-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/AntBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/HopperBulletEnv-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/InvertedPendulumBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/MinitaurBulletEnv-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/PusherBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Walker2DBulletEnv-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/HalfCheetahBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/ReacherBulletEnv-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/HumanoidBulletEnv-v0.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Pendulum-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/LunarLanderContinuous-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/MountainCarContinuous-v0.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/BipedalWalker-v3.svg"></td>
        <td></td>
    </tr>
</table>

<table>
    <tr>
        <th align="center">Benchmarked Learning Curves</th>
        <th align="center">Classic Controlt</th>
    </tr>
    <tr>
        <td align="center" colspan="2">Metrics, logs, and recorded videos are at https://app.wandb.ai/cleanrl/cleanrl.benchmark/reports/Classic-Control--VmlldzoxODE0OTQ</td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/CartPole-v1.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/Acrobot-v1.svg"></td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/LunarLander-v2.svg"></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/MountainCar-v0.svg"></td>
    </tr>
</table>


<table>
    <tr>
        <th align="center">Benchmarked Learning Curves</th>
        <th align="center">Other Experiments</th>
    </tr>
    <tr>
        <td align="center" colspan="2">Metrics, logs, and recorded videos are at https://app.wandb.ai/cleanrl/cleanrl.benchmark/reports/Others--VmlldzoxODg5ODE</td>
    </tr>
    <tr>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/BipedalWalkerHardcore-v3.svg">
        <p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</p></td>
        <td><img src="http://microrts.s3.amazonaws.com/microrts/cleanrl/open-rl-benchmark/0.3/plots/SlimeVolleySelfPlayEnv-v0.svg">
            <p>*note*: this is a self-play environment, so its episode reward should not steadily increase. Check out the video for the agent's actual performance</p></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
    </tr>
</table>


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
    --prod-mode \
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

## Support and get involved

We have a [Slack Community](https://join.slack.com/t/cleanrl/shared_invite/zt-cj64t5eq-xKZ6sD0KPGFKu1QicHEvVg) for support. Feel free to ask questions. Posting in [Github Issues](https://github.com/vwxyzjn/cleanrl/issues) and PRs are also welcome. 

In addition, we also have a monthly development cycle to implement new RL algorithms. Feel free to participate or ask questions there, too. You can sign up for our mailing list at our [Google Groups](https://groups.google.com/forum/#!forum/rlimplementation/join) to receive event RVSP which contains the Hangout video call address every week. Our past video recordings are available at [YouTube](https://www.youtube.com/watch?v=dm4HdGujpPs&list=PLQpKd36nzSuMynZLU2soIpNSMeXMplnKP&index=2)




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
            --prod-mode
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
            --prod-mode
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
