import subprocess


def test_pybullet():
    subprocess.run(
        "python cleanrl/sac_continuous_action.py --batch-size 128 --total-timesteps 135",
        shell=True,
        check=True,
    )
