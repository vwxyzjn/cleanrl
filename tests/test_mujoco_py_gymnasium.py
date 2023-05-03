import subprocess


def test_mujoco_py():
    """
    Test mujoco_py
    """
    subprocess.run(
        "python cleanrl/ddpg_continuous_action.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/ddpg_continuous_action_jax.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
