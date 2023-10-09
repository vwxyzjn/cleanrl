import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/ppo.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
