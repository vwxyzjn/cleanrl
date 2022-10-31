# export WANDB_ENTITY=openrlbenchmark

poetry install
#OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
#    --command "poetry run python cleanrl/ppo.py --cuda False --track --capture-video" \
#    --num-seeds 3 \
#    --workers 9
#
#OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
#    --command "poetry run python cleanrl/ppo_faster_1.py --cuda False --track --capture-video" \
#    --num-seeds 3 \
#    --workers 9
#
#OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
#    --command "poetry run python cleanrl/ppo_faster_2.py --cuda False --track --capture-video" \
#    --num-seeds 3 \
#    --workers 9
#
#OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
#    --command "poetry run python cleanrl/ppo_faster_3.py --cuda False --track --capture-video" \
#    --num-seeds 3 \
#    --workers 9
#
#OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
#    --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
#    --command "poetry run python cleanrl/ppo_faster_4.py --cuda False --track --capture-video" \
#    --num-seeds 3 \
#    --workers 9

OMP_NUM_THREADS=1 xvfb-run -a poetry run python -m cleanrl_utils.benchmark \
  --env-ids CartPole-v1 Acrobot-v1 MountainCar-v0 \
  --command "poetry run python cleanrl/ppo_faster_5.py --cuda False --track --capture-video" \
  --num-seeds 3 \
  --workers 9