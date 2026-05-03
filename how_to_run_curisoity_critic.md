# How To Run RND vs Curiosity-Critic

This guide runs the CleanRL RND baseline and `ppo_curiosity_critic_envpool.py` on a fresh Linux GPU workstation and compares the results in Weights & Biases.

EnvPool Atari is Linux-only in this setup. Use the same machine, same repo commit, same seeds, and same W&B project for both methods.

## 1. Prepare The Workstation

Install basic system tools:

```bash
sudo apt update
sudo apt install -y git curl tmux xvfb
```

Make sure the NVIDIA driver works:

```bash
nvidia-smi
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"
```

Clone the repo:

```bash
git clone <YOUR_CLEANRL_REPO_URL> cleanrl
cd cleanrl
```

Create and activate a Python 3.10 environment:

```bash
uv venv --python 3.10
source .venv/bin/activate
```

Install CleanRL with EnvPool support. This also installs `wandb` from the project dependencies:

```bash
uv pip install -e ".[envpool]"
```

Log in to Weights & Biases:

```bash
uv run wandb login
```

## 2. Run A Tiny Smoke Test

Before launching expensive runs, verify both scripts start and log.

```bash
TOTAL_TIMESTEPS=1000000 SEEDS="1" WANDB_PROJECT_NAME=curiosity-critic-rnd-smoke \
    bash benchmark/rnd_curiosity-critic_compare.sh
```

Check W&B for two runs:

- `ppo_rnd_envpool`, seed `1`
- `ppo_curiosity_critic_envpool`, seed `1`

If either crashes, fix that before starting full runs.

## 3. Run The Full 3-Seed Comparison

Use `tmux` so the job survives SSH disconnects:

```bash
tmux new -s rnd_cc
```

The simplest full run is the flat no-args script:

```bash
bash benchmark/rnd_curiosity-critic_compare_no_args.sh
```

This script intentionally has no loops, no environment variables, and no command-line arguments. It runs six commands one after another:

1. Curiosity-Critic seed 1
2. RND seed 1
3. Curiosity-Critic seed 2
4. RND seed 2
5. Curiosity-Critic seed 3
6. RND seed 3

It resembles `benchmark/rnd.sh` and `benchmark/curiosity_critic.sh`: each line calls `python -m cleanrl_utils.benchmark` with one seed and one worker. Because the file is no-args, it uses each training script's default W&B project name, `cleanRL`, unless you edit the six command strings directly.

For smoke tests, custom W&B project names, custom entities, or shorter debugging runs, use the configurable script instead:

```bash
WANDB_PROJECT_NAME=curiosity-critic-rnd-compare \
    bash benchmark/rnd_curiosity-critic_compare.sh
```

Optional: set a W&B entity/team:

```bash
WANDB_ENTITY=<YOUR_WANDB_ENTITY> WANDB_PROJECT_NAME=curiosity-critic-rnd-compare \
    bash benchmark/rnd_curiosity-critic_compare.sh
```

The configurable script runs six jobs one after another:

1. RND seed 1
2. Curiosity-Critic seed 1
3. RND seed 2
4. Curiosity-Critic seed 2
5. RND seed 3
6. Curiosity-Critic seed 3

Detach from `tmux` with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t rnd_cc
```

## 4. What To Watch During Training

Primary performance metric:

- `charts/avg_episodic_return`: main learning curve. Compare at equal `global_step`, not wall-clock time.

Useful runtime metric:

- `charts/SPS`: steps per second. Curiosity-Critic has an extra neural critic update, so report SPS if discussing compute.

Shared curiosity/reward metrics:

- `charts/episode_curiosity_reward`: raw intrinsic reward snapshot logged at episode end.
- `losses/fwd_loss`: RND predictor MSE for RND; WM prediction MSE for Curiosity-Critic. The update metric is matched, but the target spaces differ.

Curiosity-Critic diagnostics:

- `losses/critic_loss`: should not explode.
- `losses/error_before`: WM raw error before the minibatch WM update.
- `losses/error_after`: WM raw error after the minibatch WM update.
- `charts/critic_pred_mean`: NC baseline estimate. It should be on the same rough scale as `error_after`, not instantly collapse or blow up.

Red flags:

- `charts/avg_episodic_return` stays flat and `charts/episode_curiosity_reward` collapses near zero immediately.
- `losses/critic_loss` explodes or becomes NaN.
- `charts/SPS` is unexpectedly tiny compared with RND.
- Only one seed wins while the other two regress badly.

## 5. How To Judge The Winner

Decide the seed set before looking at results. Use the same seeds for both methods. For this comparison, use seeds `1, 2, 3`.

Primary publication plot:

- Plot `charts/avg_episodic_return` vs `global_step`.
- Show RND and Curiosity-Critic as mean curves over seeds `1, 2, 3`.
- Add shaded variability, preferably standard error or standard deviation.
- Keep smoothing identical for both methods.

Primary decision criteria:

- Curiosity-Critic should have higher mean area under the learning curve across seeds.
- Curiosity-Critic should have higher mean final performance, measured over the last stable portion of training.
- The win should not be driven by only one lucky seed.

Secondary reporting:

- Report `charts/SPS` or total wall-clock because Curiosity-Critic trains an additional NC.
- Report auxiliary parameter counts:
  - RND predictor trainable params: about `2.203M`
  - Curiosity-Critic WM trainable params: about `2.203M`
  - RND target params: about `1.678M` frozen
  - Curiosity-Critic NC trainable params: about `1.678M`

Recommended conclusion rule:

- If Curiosity-Critic beats RND in both mean AUC and mean final return across all three seeds, it is the likely winner.
- If curves are mixed, report it honestly and run more seeds before making a strong publication claim.
- If Curiosity-Critic is better in environment steps but much slower in wall-clock, describe it as a sample-efficiency improvement and include compute caveats.

## 6. Useful One-Off Commands

Run only seed 1 for both methods:

```bash
SEEDS="1" bash benchmark/rnd_curiosity-critic_compare.sh
```

Run a shorter debugging job:

```bash
TOTAL_TIMESTEPS=5000000 SEEDS="1" bash benchmark/rnd_curiosity-critic_compare.sh
```

Run only RND manually:

```bash
uv run python cleanrl/ppo_rnd_envpool.py \
    --env-id MontezumaRevenge-v5 \
    --seed 1 \
    --track \
    --wandb-project-name curiosity-critic-rnd-compare
```

Run only Curiosity-Critic manually:

```bash
uv run python cleanrl/ppo_curiosity_critic_envpool.py \
    --env-id MontezumaRevenge-v5 \
    --seed 1 \
    --track \
    --wandb-project-name curiosity-critic-rnd-compare
```
