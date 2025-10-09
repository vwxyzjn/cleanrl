import subprocess


def test_ppo_multigpu():
    subprocess.run(
        "torchrun --standalone --nnodes=1 --nproc_per_node=2 cleanrl/ppo_atari_multigpu.py --num-envs 8 --num-steps 32 --total-timesteps 256",
        shell=True,
        check=True,
    )
