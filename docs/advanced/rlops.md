# RLops

This document describes how to we do "RLops" to validate new features / bug fixes and avoid introducing regressions.


## Background
DRL is brittle and has a series of reproducibility issues — even bug fixes sometimes could introduce performance regression (e.g., see [how a bug fix of contact force in MuJoCo results in worse performance for PPO](https://github.com/openai/gym/pull/2762#discussion_r853488897)). Therefore, it is essential to understand how the proposed changes impact the performance of the algorithms. At large, we wish to distinguish two types of contributions: 1) **non-performance-impacting changes** and 2) **performance-impacting changes**. 

* **non-performance-impacting changes**: this type of change does *not* impact the performance of the algorithm, such as documentation fixes (#282), renaming variables (#257), and removing unused code (#287). For this type of change, we can easily merge them without worrying too much about the consequences.
* **performance-impacting changes**: this type of change impacts the algorithm's performance. Examples include making a slight modification to the `gamma` parameter in PPO  (https://github.com/vwxyzjn/cleanrl/pull/209), properly handling action bounds in DDPG (https://github.com/vwxyzjn/cleanrl/pull/211), and fixing bugs (https://github.com/vwxyzjn/cleanrl/pull/281)


**Importantly, regardless of the slight difference in performance-impacting changes, we need to re-run the benchmark to ensure there is no regression**. This post proposes a way for us to re-run the model and check regression seamlessly.

## Methodology


### (Step 1) Run the benchmark

Given a new feature, we create a PR and then run the benchmark experiments through [`benchmark.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py), such as the following:

```bash
poetry install --with mujoco,pybullet
python -c "import mujoco_py"
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --command "poetry run python cleanrl/ddpg_continuous_action.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1
```

under the hood, this script will invoke an `--autotag` feature that tries to tag the the experiments with version control information, such as the git tag (e.g., `v1.0.0b2-8-g6081d30`) and the github PR number (e.g., `pr-299`). This is useful for us to compare the performance of the same algorithm across different versions.


### (Step 2) Regression check

Let's say our latest experiments is tagged with `pr-299`. We can then run the following command to compare its performance with our pilot experiments `rlops-pilot`. Note that the pilot experiments include all experiments before we started using RLops (i.e., `rlops-pilot` is the baseline).


```bash
python -m cleanrl_utils.rlops --exp-name ddpg_continuous_action \
    --wandb-project-name cleanrl \
    --wandb-entity openrlbenchmark \
    --tags 'pr-299' 'rlops-pilot' \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --output-filename compare.png \
    --scan-history \
    --report
```
```
CleanRL's ddpg_continuous_action (pr-299) in HalfCheetah-v2 has 3 runs
┣━━ HalfCheetah-v2__ddpg_continuous_action__4__1667280971 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ HalfCheetah-v2__ddpg_continuous_action__3__1667271574 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ HalfCheetah-v2__ddpg_continuous_action__2__1667261986 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
CleanRL's ddpg_continuous_action (pr-299) in Walker2d-v2 has 3 runs
┣━━ Walker2d-v2__ddpg_continuous_action__4__1667284233 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ Walker2d-v2__ddpg_continuous_action__3__1667274709 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ Walker2d-v2__ddpg_continuous_action__2__1667265261 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
CleanRL's ddpg_continuous_action (pr-299) in Hopper-v2 has 3 runs
┣━━ Hopper-v2__ddpg_continuous_action__4__1667287363 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ Hopper-v2__ddpg_continuous_action__3__1667277826 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
┣━━ Hopper-v2__ddpg_continuous_action__2__1667268434 with tags = ['pr-299', 'v1.0.0b2-8-g6081d30']
CleanRL's ddpg_continuous_action (rlops-pilot) in HalfCheetah-v2 has 3 runs
┣━━ HalfCheetah-v2__ddpg_continuous_action__3__1651008691 with tags = ['latest', 'rlops-pilot']
┣━━ HalfCheetah-v2__ddpg_continuous_action__2__1651004631 with tags = ['latest', 'rlops-pilot']
┣━━ HalfCheetah-v2__ddpg_continuous_action__1__1651000539 with tags = ['latest', 'rlops-pilot']
CleanRL's ddpg_continuous_action (rlops-pilot) in Walker2d-v2 has 3 runs
┣━━ Walker2d-v2__ddpg_continuous_action__3__1651008768 with tags = ['latest', 'rlops-pilot']
┣━━ Walker2d-v2__ddpg_continuous_action__2__1651004640 with tags = ['latest', 'rlops-pilot']
┣━━ Walker2d-v2__ddpg_continuous_action__1__1651000539 with tags = ['latest', 'rlops-pilot']
CleanRL's ddpg_continuous_action (rlops-pilot) in Hopper-v2 has 3 runs
┣━━ Hopper-v2__ddpg_continuous_action__3__1651008797 with tags = ['latest', 'rlops-pilot']
┣━━ Hopper-v2__ddpg_continuous_action__2__1651004715 with tags = ['latest', 'rlops-pilot']
┣━━ Hopper-v2__ddpg_continuous_action__1__1651000539 with tags = ['latest', 'rlops-pilot']
               CleanRL's ddpg_continuous_action (pr-299) CleanRL's ddpg_continuous_action (rlops-pilot)
HalfCheetah-v2                         10323.36 ± 112.39                               9327.00 ± 161.20
Walker2d-v2                            1841.98 ± 1240.35                               1173.64 ± 612.88
Hopper-v2                               1000.18 ± 636.04                               1167.50 ± 962.93
```


which could generate the table above, which reports the mean and standard deviation of the performance of the algorithm in the last 20 episodes. 

the following image and a wandb report.

![](./compare.png)

<iframe loading="lazy" src="Regression Report: ddpg_continuous_action (['pr-299', 'rlops-pilot'])" style="width:100%; height:500px" title="MuJoCo: CleanRL's DDPG + JAX"></iframe>



!!! info+

    **Support for multiple tags, their inclusions and exclusions, and filter by users**: The syntax looks like `--tags "tag1;tag2!tag3;tag4?user1"`, where tag1 and tag2 are included, tag3 and tag4 are excluded, and user1 is included. Here are some examples:

    ```bash
    python -m cleanrl_utils.rlops --exp-name ddpg_continuous_action_jax \
        --wandb-project-name cleanrl \
        --wandb-entity openrlbenchmark \
        --tags 'pr-298?costa-huang' 'rlops-pilot?costa-huang' \
        --env-ids Hopper-v2 Walker2d-v2 HalfCheetah-v2 \
        --output-filename compare.png \
        --report

    python -m cleanrl_utils.rlops --exp-name ddpg_continuous_action_jax \
        --wandb-project-name cleanrl \
        --wandb-entity openrlbenchmark \
        --tags 'pr-298?joaogui1' 'rlops-pilot?joaogui1' \
        --env-ids Hopper-v2 Walker2d-v2 HalfCheetah-v2 \
        --output-filename compare.png \
        --report
    ```


!!! warning

    The `-m cleanrl_utils.rlops` script is still in its early stage. Please feel free to open an issue if you have any questions or suggestions.

### (Step 3) Merge the PR

Once we confirm there is no regression in the performance, we can merge the PR. Furthermore, we will label the new experiments as `latest` (and remove the tag `latest` for `v1.0.0b2-7-gxfd3d3` correspondingly. 

```bash
python -m -m cleanrl_utils.rlops_tags --add latest --source-tag v1.0.0b2-9-g4605546
python -m -m cleanrl_utils.rlops_tags --remove latest --source-tag rlops-pilot
```
