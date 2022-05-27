import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/ppo_procgen.py --num-envs 1 --num-steps 64 --total-timesteps 256 --num-minibatches 1",
        shell=True,
        check=True,
    )


def test_ppo():
    subprocess.run(
        "python cleanrl/ppg_procgen.py --num-envs 1 --num-steps 64 --total-timesteps 256 --num-minibatches 1 --n-iteration 1",
        shell=True,
        check=True,
    )
