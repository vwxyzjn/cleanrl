import subprocess


def test_ppo_envpool():
    subprocess.run(
        "python cleanrl/ppo_atari_envpool.py --num-envs 8 --num-steps 32 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_envpool():
    subprocess.run(
        "python cleanrl/ppo_rnd_envpool.py --num-envs 8 --num-steps 32 --num-iterations-obs-norm-init 1 --total-timesteps 256",
        shell=True,
        check=True,
    )
