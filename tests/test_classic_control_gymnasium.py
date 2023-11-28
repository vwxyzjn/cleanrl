import subprocess


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51():
    subprocess.run(
        "python cleanrl/c51.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_eval():
    subprocess.run(
        "python cleanrl/c51.py --save-model --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
