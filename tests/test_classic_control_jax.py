import subprocess


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/dqn_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
