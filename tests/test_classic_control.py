import subprocess

def test_classic_control():
    """
    Test classic control
    """
    subprocess.run("python cleanrl/ppo.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/dqn.py --total-timesteps 1000", shell=True, check=True)
    subprocess.run("python cleanrl/c51.py --total-timesteps 1000", shell=True, check=True)
