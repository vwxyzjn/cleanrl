This folder contains experimental script for brax

python old_ppo_that_works.py --capture-video --prod-mode --wandb-project brax --gym-id HalfCheetahBulletEnv-v0
python ppo_continuous_action.py --capture-video --prod-mode --wandb-project brax --gym-id HalfCheetahBulletEnv-v0 --cuda False