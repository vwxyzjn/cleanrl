import subprocess


def test_mujoco():
    """
    Test mujoco
    """
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --env-id Hopper-v4 --num-envs 1 --num-steps 64 --total-timesteps 128",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/ppo_continuous_action.py --env-id dm_control/cartpole-balance-v0 --num-envs 1 --num-steps 64 --total-timesteps 128",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/td3_continuous_action_jax.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/td3_continuous_action.py --env-id Hopper-v4 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/rpo_continuous_action.py --env-id Hopper-v4 --num-envs 1 --num-steps 64 --total-timesteps 128",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/rpo_continuous_action.py --env-id dm_control/cartpole-balance-v0 --num-envs 1 --num-steps 64 --total-timesteps 128",
        shell=True,
        check=True,
    )
