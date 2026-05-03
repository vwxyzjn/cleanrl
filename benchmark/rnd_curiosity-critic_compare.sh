#!/usr/bin/env bash
set -euo pipefail

# Sequential RND vs Curiosity-Critic comparison.
# Run from the repository root:
#   bash benchmark/rnd_curiosity-critic_compare.sh
#
# Defaults run MontezumaRevenge-v5 for seeds 1, 2, and 3 with each script's
# default total timesteps. Override these environment variables for smoke tests
# or alternate W&B routing:
#   TOTAL_TIMESTEPS=1000000 SEEDS="1" bash benchmark/rnd_curiosity-critic_compare.sh
#   WANDB_PROJECT_NAME=my-project WANDB_ENTITY=my-team bash benchmark/rnd_curiosity-critic_compare.sh

ENV_ID="${ENV_ID:-MontezumaRevenge-v5}"
SEEDS="${SEEDS:-1 2 3}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-curiosity-critic-rnd-compare}"

# Install project dependencies, including envpool and wandb, into the active uv environment.
uv pip install ".[envpool]"

run_cleanrl() {
    local label="$1"
    local script="$2"
    local seed="$3"

    local command=(
        uv run python "${script}"
        --env-id "${ENV_ID}"
        --seed "${seed}"
        --track
        --wandb-project-name "${WANDB_PROJECT_NAME}"
    )

    if [[ -n "${WANDB_ENTITY:-}" ]]; then
        command+=(--wandb-entity "${WANDB_ENTITY}")
    fi

    if [[ -n "${TOTAL_TIMESTEPS:-}" ]]; then
        command+=(--total-timesteps "${TOTAL_TIMESTEPS}")
    fi

    echo "======================================================================"
    echo "Starting ${label}: env=${ENV_ID}, seed=${seed}, project=${WANDB_PROJECT_NAME}"
    echo "Command: ${command[*]}"
    echo "======================================================================"

    # EnvPool Atari runs are commonly launched under xvfb on headless machines.
    if command -v xvfb-run >/dev/null 2>&1; then
        xvfb-run -a "${command[@]}"
    else
        "${command[@]}"
    fi
}

for seed in ${SEEDS}; do
    # Baseline first for this seed.
    run_cleanrl "RND" "cleanrl/ppo_rnd_envpool.py" "${seed}"

    # Curiosity-Critic second for the same seed.
    run_cleanrl "Curiosity-Critic" "cleanrl/ppo_curiosity_critic_envpool.py" "${seed}"
done

echo "All requested RND and Curiosity-Critic runs completed."
