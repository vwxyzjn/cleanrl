export WANDB_ENTITY=openrlbenchmark

poetry install -E "mujoco pybullet"
python -c "import mujoco_py"

#xvfb-run -a python -m cleanrl_utils.benchmark \
#    --env-ids Swimmer-v3 \
#    --command "poetry run python cleanrl/apo_continuous_action.py --track --capture-video --gae-lambda=0.99" \
#    --num-seeds 3 \
#    --workers 3

#xvfb-run -a python -m cleanrl_utils.benchmark \
#    --env-ids HalfCheetah-v3 \
#    --command "poetry run python cleanrl/apo_continuous_action.py --track --capture-video --gae-lambda=0.9" \
#    --num-seeds 3 \
#    --workers 3

#xvfb-run -a python -m cleanrl_utils.benchmark \
#    --env-ids Ant-v3 \
#    --command "poetry run python cleanrl/apo_continuous_action.py --track --capture-video --gae-lambda=0.9" \
#    --num-seeds 3 \
#    --workers 3

xvfb-run -a python -m cleanrl_utils.benchmark \
    --env-ids Hopper-v3 \
    --command "poetry run python cleanrl/apo_continuous_action.py --track --capture-video --gae-lambda=0.99" \
    --num-seeds 3 \
    --workers 3