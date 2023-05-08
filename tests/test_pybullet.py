import subprocess


def test_pybullet():
    subprocess.run(
        "python cleanrl/td3_continuous_action.py --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
