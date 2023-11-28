import subprocess


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/dqn_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_dqn_jax_eval():
    subprocess.run(
        "python cleanrl/dqn_atari_jax.py --save-model --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_qdagger_dqn_atari_jax_impalacnn():
    subprocess.run(
        "python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1",
        shell=True,
        check=True,
    )


def test_qdagger_dqn_atari_jax_impalacnn_eval():
    subprocess.run(
        "python cleanrl/qdagger_dqn_atari_jax_impalacnn.py --save-model --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4 --teacher-steps 16 --offline-steps 16 --teacher-eval-episodes 1",
        shell=True,
        check=True,
    )


def test_c51_atari_jax():
    subprocess.run(
        "python cleanrl/c51_atari_jax.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_c51_atari_jax_eval():
    subprocess.run(
        "python cleanrl/c51_atari_jax.py --save-model --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )
