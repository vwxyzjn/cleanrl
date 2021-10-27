# Basic Usage

## Two Ways to Run
After the dependencies have been installed, there are **two ways** to run 
the CleanRL script under the poetry virtual environments.


1. Using `poetry run`:

    ```bash
    poetry run python cleanrl/ppo.py \
        --seed 1 \
        --gym-id CartPole-v0 \
        --total-timesteps 50000
    ```
    <script id="asciicast-443649" src="https://asciinema.org/a/443649.js" async></script>


2. Using `poetry shell`:

    1. We first activate the virtual environment by using
    `poetry shell`
    2. Then, run any desired CleanRL script
   
    Attention: Each step must be executed separately!


    ```bash
    poetry shell
    ```
    ```bash
    python cleanrl/ppo.py \
        --seed 1 \
        --gym-id CartPole-v0 \
        --total-timesteps 50000
    ```
    <script id="asciicast-JL1FR00I2JNklAhMd2dwEAQuz" src="https://asciinema.org/a/JL1FR00I2JNklAhMd2dwEAQuz.js" async></script>

!!! note

    We recommend `poetry shell` workflow for development. When the shell is activeated, you should
    be seeing a prefix like `(cleanrl-iXg02GqF-py3.9)` in your shell's prompt, which is the name
    of the poetry's virtual environment.
    **We will assume to run other commands (e.g. `tensorboard`) in the documentation within the poetry's shell.**


## Visualize Training Metrics

By default, the CleanRL scripts record all the training metrics via Tensorboard
into the `runs` folder. So, after running the training script above, feel free to run

```bash
tensorboard --logdir runs
```

![Tensorboard](tensorboard.png)


## Visualize the Agent's Gameplay Videos

CleanRL helps record the agent's gameplay videos with a `--capture-video` flag,
which will save the videos in the `videos/{$run_name}` folder.

```bash linenums="1" hl_lines="5"
python cleanrl/ppo.py \
    --seed 1 \
    --gym-id CartPole-v0 \
    --total-timesteps 50000 \
    --capture-video
```

![videos](videos.png)
![videos2](videos2.png)