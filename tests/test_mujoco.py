import subprocess


def test_mujoco():
    """
    Test mujoco
    """
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --env-id Hopper-v4 --num-envs 1 --num-steps 32 --total-timesteps 128",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --env-id dm_control/cartpole-balance-v0 --num-envs 1 --num-steps 32 --total-timesteps 128",
        shell=True,
        check=True,
    )
