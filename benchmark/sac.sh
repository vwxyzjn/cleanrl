poetry install --with mujoco_py,pybullet
poetry run python -c "import mujoco_py"
OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
    --env-ids HalfCheetah-v2 Walker2d-v2 Hopper-v2 \
    --command "poetry run python cleanrl/sac_continuous_action.py --track --capture-video" \
    --num-seeds 3 \
    --workers 3