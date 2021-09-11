This folder contains experimental script for brax

```bash
python old_ppo_that_works.py --capture-video --track --wandb-project brax --gym-id HalfCheetahBulletEnv-v0
python ppo_continuous_action.py --capture-video --track --wandb-project brax --gym-id HalfCheetahBulletEnv-v0 --cuda False
python ppo_brax_test.py --track --wandb-project brax --cuda False
```
