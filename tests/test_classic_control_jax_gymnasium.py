import subprocess


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/dqn_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_jax():
    subprocess.run(
        "python cleanrl/c51_jax.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51_jax_eval():
    subprocess.run(
        "python cleanrl/c51_jax.py --save-model --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
