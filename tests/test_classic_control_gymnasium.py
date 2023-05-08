import subprocess


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
