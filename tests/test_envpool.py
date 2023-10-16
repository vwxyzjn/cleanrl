import subprocess


def test_ppo_atari_envpool():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool.py --num-envs 8 --num-steps 32 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_rnd_envpool():
    subprocess.run(
        "python cleanrl/ppo_rnd_envpool.py --num-envs 8 --num-steps 32 --num-iterations-obs-norm-init 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_atari_envpool_xla_jax_scan_eval():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool_xla_jax_scan.py --save-model --num-envs 8 --num-steps 6 --update-epochs 1 --num-minibatches 1 --total-timesteps 256",
        shell=True,
        check=True,
    )
