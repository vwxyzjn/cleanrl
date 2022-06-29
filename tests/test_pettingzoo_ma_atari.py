import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/ppo_pettingzoo_ma_atari.py --num-steps 32 --num-envs 6 --total-timesteps 256",
        shell=True,
        check=True,
    )
