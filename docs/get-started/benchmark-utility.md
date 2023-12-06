# Benchmark Utility

CleanRL comes with a utility module `cleanrl_utils.benchmark` to help schedule and run benchmark experiments on your local machine.

## Usage

Try running `python -m cleanrl_utils.benchmark --help` to get the help text.

```bash
$ python -m cleanrl_utils.benchmark --help
usage: benchmark.py [-h] --env-ids [STR
                    [STR ...]] --command STR [--num-seeds INT]
                    [--start-seed INT] [--workers INT]
                    [--auto-tag | --no-auto-tag]
                    [--slurm-template-path {None}|STR]
                    [--slurm-gpus-per-task {None}|INT]
                    [--slurm-total-cpus {None}|INT]
                    [--slurm-ntasks {None}|INT] [--slurm-nodes {None}|INT]

╭─ arguments ──────────────────────────────────────────────────────────────╮
│ -h, --help                                                               │
│     show this help message and exit                                      │
│ --env-ids [STR [STR ...]]                                                │
│     the ids of the environment to compare (required)                     │
│ --command STR                                                            │
│     the command to run (required)                                        │
│ --num-seeds INT                                                          │
│     the number of random seeds (default: 3)                              │
│ --start-seed INT                                                         │
│     the number of the starting seed (default: 1)                         │
│ --workers INT                                                            │
│     the number of workers to run benchmark experimenets (default: 0)     │
│ --auto-tag, --no-auto-tag                                                │
│     if toggled, the runs will be tagged with git tags, commit, and pull  │
│     request number if possible (default: True)                           │
│ --slurm-template-path {None}|STR                                         │
│     the path to the slurm template file (see docs for more details)      │
│     (default: None)                                                      │
│ --slurm-gpus-per-task {None}|INT                                         │
│     the number of gpus per task to use for slurm jobs (default: None)    │
│ --slurm-total-cpus {None}|INT                                            │
│     the number of gpus per task to use for slurm jobs (default: None)    │
│ --slurm-ntasks {None}|INT                                                │
│     the number of tasks to use for slurm jobs (default: None)            │
│ --slurm-nodes {None}|INT                                                 │
│     the number of nodes to use for slurm jobs (default: None)            │
╰──────────────────────────────────────────────────────────────────────────╯
```

## Examples

The following example demonstrates how to run classic control benchmark experiments.

```bash
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --no_cuda --track --capture_video" \
    --num-seeds 3 \
    --workers 5
```

What just happened here? In principle the helps run the following commands in 5 subprocesses:

```bash
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id CartPole-v1 --seed 1
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id Acrobot-v1 --seed 1
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id MountainCar-v0 --seed 1
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id CartPole-v1 --seed 2
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id Acrobot-v1 --seed 2
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id MountainCar-v0 --seed 2
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id CartPole-v1 --seed 3
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id Acrobot-v1 --seed 3
poetry run python cleanrl/ppo.py --no_cuda --track --capture_video --env-id MountainCar-v0 --seed 3
```

More specifically:

1. `--env-ids CartPole-v1 Acrobot-v1 MountainCar-v0` specifies that running experiments against these three environments
1. `--command "poetry run python cleanrl/ppo.py --no_cuda --track --capture_video"` suggests running `ppo.py` with these settings:
    * turn off GPU usage via `--no_cuda`: because `ppo.py` has such as small neural network it often runs faster on CPU only
    * track the experiments via `--track`
    * render the agent gameplay videos via `--capture_video`; these videos algo get saved to the tracked experiments
        * ` xvfb-run -a` virtualizes a display for video recording, enabling these commands on a headless linux system
1. `--num-seeds 3` suggests running the the command with 3 random seeds for each `env-id`
1. `--workers 5` suggests at maximum using 5 subprocesses to run the experiments
    * `OMP_NUM_THREADS=1` suggests `torch` to use only 1 thread for each subprocesses; this way we don't have processes fighting each other.
1. `--autotag` tries to tag the the experiments with version control information, such as the git tag (e.g., `v1.0.0b2-8-g6081d30`) and the github PR number (e.g., `pr-299`). This is useful for us to compare the performance of the same algorithm across different versions.


Note that when you run with high-throughput environments such as `envpool` or `procgen`, it's recommended to set `--workers 1` to maximuize SPS (steps per second), such as

```bash
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool.py --track --capture_video" \
    --num-seeds 3 \
    --workers 1
```

For more example usage, see [https://github.com/vwxyzjn/cleanrl/blob/master/benchmark](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark)


## Slurm integration

If you have access to a slurm cluster, you can use `cleanrl_utils.benchmark` to schedule jobs on the cluster. The following example demonstrates how to run classic control benchmark experiments on a slurm cluster.

``` title="benchmark/ppo.sh" linenums="1"
--8<-- "benchmark/ppo.sh:3:12"
```

```
poetry install
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --no_cuda --track --capture_video" \
    --num-seeds 3 \
    --workers 9 \
    --slurm-gpus-per-task 1 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 10 \
    --slurm-template-path benchmark/cleanrl_1gpu.slurm_template
```

Here, we have
* `--slurm-gpus-per-task 1` suggests that each slurm job should use 1 GPU
* `--slurm-ntasks 1` suggests that each slurm job should use 1 CPU
* `--slurm-total-cpus 10` suggests that each slurm job should use 10 CPUs in total
* `--slurm-template-path benchmark/cleanrl_1gpu.slurm_template` suggests that we should use the template file `benchmark/cleanrl_1gpu.slurm_template` to generate the slurm job scripts. The template file looks like this:

``` title="benchmark/cleanrl_1gpu.slurm_template" linenums="1"
--8<-- "benchmark/cleanrl_1gpu.slurm_template"
```

The utility will generate a slurm script based on the template file and submit the job to the cluster. The generated slurm script looks like this:

```
#!/bin/bash
#SBATCH --job-name=low-priority
#SBATCH --partition=production-cluster
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --array=0-8%9
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=ip-26-0-147-[245,247],ip-26-0-156-239
##SBATCH --nodelist=ip-26-0-156-13


env_ids=(CartPole-v1 Acrobot-v1 MountainCar-v0)
seeds=(1 2 3)
env_id=${env_ids[$SLURM_ARRAY_TASK_ID / 3]}
seed=${seeds[$SLURM_ARRAY_TASK_ID % 3]}

echo "Running task $SLURM_ARRAY_TASK_ID with env_id: $env_id and seed: $seed"

srun poetry run python cleanrl/ppo.py --no_cuda --track --env-id $env_id --seed $seed # 
```

