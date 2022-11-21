# Benchmark Utility

CleanRL comes with a utility module `cleanrl_utils.benchmark` to help schedule and run benchmark experiments on your local machine.

## Usage

Try running `python -m cleanrl_utils.benchmark --help` to get the help text.

```bash
python -m cleanrl_utils.benchmark --help
usage: benchmark.py [-h] [--env-ids ENV_IDS [ENV_IDS ...]] [--command COMMAND] [--num-seeds NUM_SEEDS] [--start-seed START_SEED] [--workers WORKERS]
                    [--auto-tag [AUTO_TAG]]

optional arguments:
  -h, --help            show this help message and exit
  --env-ids ENV_IDS [ENV_IDS ...]
                        the ids of the environment to benchmark
  --command COMMAND     the command to run
  --num-seeds NUM_SEEDS
                        the number of random seeds
  --start-seed START_SEED
                        the number of the starting seed
  --workers WORKERS     the number of workers to run benchmark experimenets
  --auto-tag [AUTO_TAG]
                        if toggled, the runs will be tagged with git tags, commit, and pull request number if possible
```

## Examples

The following example demonstrates how to run classic control benchmark experiments.

```bash
OMP_NUM_THREADS=1 xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
    --num-seeds 3 \
    --workers 5
```

What just happened here? In principle the helps run the following commands in 5 subprocesses:

```bash
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id CartPole-v1 --seed 1
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id Acrobot-v1 --seed 1
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id MountainCar-v0 --seed 1
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id CartPole-v1 --seed 2
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id Acrobot-v1 --seed 2
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id MountainCar-v0 --seed 2
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id CartPole-v1 --seed 3
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id Acrobot-v1 --seed 3
poetry run python cleanrl/ppo.py --cuda False --track --capture-video --env-id MountainCar-v0 --seed 3
```

More specifically:

1. `--env-ids CartPole-v1 Acrobot-v1 MountainCar-v0` specifies that running experiments against these three environments
1. `--command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video"` suggests running `ppo.py` with these settings:
    * turn off GPU usage via `--cuda False`: because `ppo.py` has such as small neural network it often runs faster on CPU only
    * track the experiments via `--track`
    * render the agent gameplay videos via `--capture-video`; these videos algo get saved to the tracked experiments
        * ` xvfb-run -a` virtualizes a display for video recording, enabling these commands on a headless linux system
1. `--num-seeds 3` suggests running the the command with 3 random seeds for each `env-id`
1. `--workers 5` suggests at maximum using 5 subprocesses to run the experiments
    * `OMP_NUM_THREADS=1` suggests `torch` to use only 1 thread for each subprocesses; this way we don't have processes fighting each other.
1. `--autotag` tries to tag the the experiments with version control information, such as the git tag (e.g., `v1.0.0b2-8-g6081d30`) and the github PR number (e.g., `pr-299`). This is useful for us to compare the performance of the same algorithm across different versions.


Note that when you run with high-throughput environments such as `envpool` or `procgen`, it's recommended to set `--workers 1` to maximuize SPS (steps per second), such as

```bash
xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids Pong-v5 BeamRider-v5 Breakout-v5 \
    --command "poetry run python cleanrl/ppo_atari_envpool.py --track --capture-video" \
    --num-seeds 3 \
    --workers 1
```

For more example usage, see [https://github.com/vwxyzjn/cleanrl/blob/master/benchmark](https://github.com/vwxyzjn/cleanrl/blob/master/benchmark)