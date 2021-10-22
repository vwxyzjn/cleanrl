import subprocess

def test_pybullet():
    """
    Test classic control
    """
    subprocess.run("python cleanrl/ppo_continuous_action.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/ddpg_continuous_action.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/td3_continuous_action.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/sac_continuous_action.py --total-timesteps 1000", shell=True, check=True)
