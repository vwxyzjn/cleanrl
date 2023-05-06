import subprocess


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_dqn_eval():
    subprocess.run(
        "python cleanrl/dqn_atari.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )