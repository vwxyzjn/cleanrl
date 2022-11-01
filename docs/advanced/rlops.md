# RLops

This document describes how to we do "RLops" to validate new features / bug fixes and avoid introducing regressions.


## Background
DRL is brittle and has a series of reproducibility issues — even bug fixes sometimes could introduce performance regression (e.g., see [how a bug fix of contact force in MuJoCo results in worse performance for PPO](https://github.com/openai/gym/pull/2762#discussion_r853488897)). Therefore, it is essential to understand how the proposed changes impact the performance of the algorithms. At large, we wish to distinguish two types of contributions: 1) **non-performance-impacting changes** and 2) **performance-impacting changes**. 

* **non-performance-impacting changes**: this type of change does *not* impact the performance of the algorithm, such as documentation fixes (#282), renaming variables (#257), and removing unused code (#287). For this type of change, we can easily merge them without worrying too much about the consequences.
* **performance-impacting changes**: this type of change impacts the algorithm's performance. Examples include making a slight modification to the `gamma` parameter in PPO  (https://github.com/vwxyzjn/cleanrl/pull/209), properly handling action bounds in DDPG (https://github.com/vwxyzjn/cleanrl/pull/211), and fixing bugs (https://github.com/vwxyzjn/cleanrl/pull/281)


**Importantly, regardless of the slight difference in performance-impacting changes, we need to re-run the benchmark to ensure there is no regression**. This post proposes a way for us to re-run the model and check regression seamlessly.

## Methodology


### (Step 1) Run the benchmark

We usually ran the benchmark experiments through [`benchmark.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl_utils/benchmark.py), such as the following:

```bash
poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 9
```

under the hood, this script will invoke an `autotag` feature that tries to tag the the experiments with version control information, such as the git tag (e.g., `v1.0.0b1-4-g4ea73d9`) and the github PR number (e.g., `pr-308`). This is useful for us to compare the performance of the same algorithm across different versions.


### (Step 2) Regression check

Let's say our latest experiments is tagged with `v1.0.0b2-9-g4605546`. We can then run the following command to compare its performance with the the current version `latest`:


```bash
python rlops.py --exp-name ddpg_continuous_action_jax \
    --wandb-project-name cleanrl \
    --wandb-entity openrlbenchmark \
    --tags v1.0.0b2-9-g4605546 rlops-pilot \
    --env-ids Hopper-v2 Walker2d-v2 HalfCheetah-v2 \
    --output-filename compare.png \
    --report
```
which could generate wandb reports with the following figure and corresponding tables.

<img width="1195" alt="image" src="https://user-images.githubusercontent.com/5555347/196775462-2ef25c47-72dd-426d-88b8-9d74e5062936.png">


### (Step 3) Merge the PR

Once we confirm there is no regression in the performance, we can merge the PR. Furthermore, we will label the new experiments as `latest` (and remove the tag `latest` for `v1.0.0b2-7-gxfd3d3` correspondingly. 

```bash
python rlops_tags.py --add latest --source-tag v1.0.0b2-9-g4605546
python rlops_tags.py --remove latest --source-tag rlops-pilot
```
```
