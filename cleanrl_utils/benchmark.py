import argparse
import shlex
import subprocess


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-ids", nargs="+", default=["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        help="the ids of the environment to benchmark")
    parser.add_argument("--command", type=str, default="poetry run python cleanrl/ppo.py",
        help="the command to run")
    parser.add_argument("--num-seeds", type=int, default=3,
        help="the number of random seeds")
    parser.add_argument('--workers', type=int, default=0,
        help='the number of eval workers to run benchmark experimenets (skips evaluation when set to 0)')
    args = parser.parse_args()
    # fmt: on
    return args


def run_experiment(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list)
    return_code = fd.wait()
    assert return_code == 0


if __name__ == "__main__":
    args = parse_args()
    commands = []
    for seed in range(1, args.num_seeds + 1):
        for env_id in args.env_ids:
            commands += [" ".join([args.command, "--env-id", env_id, "--seed", str(seed)])]

    print(commands)

    if args.workers > 0:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="cleanrl-benchmark-worker-")
        for command in commands:
            executor.submit(run_experiment, command)
        executor.shutdown(wait=True, cancel_futures=False)
