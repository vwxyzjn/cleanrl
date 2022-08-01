import subprocess


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/dqn_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )
