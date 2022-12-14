import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/ppo_atari.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_lstm():
    subprocess.run(
        "python cleanrl/ppo_atari_lstm.py --num-envs 4 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn():
    subprocess.run(
        "python cleanrl/dqn_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )


def test_c51():
    subprocess.run(
        "python cleanrl/c51_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )

def test_dreamer_disc_latents():
    subprocess.run(
        "python cleanrl/dreamer_atari.py --buffer-prefill 500 --total-timesteps 512 --buffer-size 2000 --batch-size 4 --batch-length 50 --train-every 4",
        shell=True,
        check=True,
    )

def test_dreamer_cont_latents():
    subprocess.run(
        "python cleanrl/dreamer_atari.py --buffer-prefill 400 --total-timesteps 450 --buffer-size 2000 --batch-size 4 --batch-length 50 --train-every 4 --rssm-discrete 0",
        shell=True,
        check=True,
    )