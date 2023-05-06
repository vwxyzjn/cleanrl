import subprocess


def test_dqn():
    subprocess.run(
        "python enjoy.py --exp-name dqn --env CartPole-v1 --eval-episodes 1",
        shell=True,
        check=True,
    )


def test_dqn_atari():
    subprocess.run(
        "python enjoy.py --exp-name dqn_atari --env BreakoutNoFrameskip-v4 --eval-episodes 1",
        shell=True,
        check=True,
    )


def test_dqn_jax():
    subprocess.run(
        "python enjoy.py --exp-name dqn_jax --env CartPole-v1 --eval-episodes 1",
        shell=True,
        check=True,
    )
