import subprocess


def test_pybullet():
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/ddpg_continuous_action.py --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/td3_continuous_action.py --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/sac_continuous_action.py --batch-size 128 --total-timesteps 135",
        shell=True,
        check=True,
    )
