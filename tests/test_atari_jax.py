import subprocess


def test_c51_jax():
    subprocess.run(
        "python cleanrl/c51_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_c51_jax_eval():
    subprocess.run(
        "python cleanrl/c51_atari_jax.py --save-model True --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )
