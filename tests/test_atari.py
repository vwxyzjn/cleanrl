import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/ppo_atari.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn_atari.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51():
    subprocess.run(
        "python cleanrl/c51_atari.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
