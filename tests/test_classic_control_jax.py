import subprocess


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/dqn_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_sac_jax():
    subprocess.run(
        "python cleanrl/sac_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
