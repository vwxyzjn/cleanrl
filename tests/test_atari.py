import subprocess

def test_atari():
    """
    Test classic control
    """
    subprocess.run("python cleanrl/ppo_atari.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/dqn_atari.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/c51_atari.py --total-timesteps 1000", shell=True, check=True)
