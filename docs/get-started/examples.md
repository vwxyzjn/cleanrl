# Examples

## Atari
```
poetry shell

poetry install -E atari
python cleanrl/dqn_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/c51_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/apex_dqn_atari.py --gym-id BreakoutNoFrameskip-v4

# NEW: 3-4x side-effects free speed up with envpool's atari (only available to linux)
poetry install -E procgen
python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4
```
### Demo

<script id="asciicast-443625" src="https://asciinema.org/a/443625.js" async></script>

You can also run training scripts in other games, such as:

## Classic Control
```
poetry shell

python cleanrl/dqn.py --gym-id CartPole-v1
python cleanrl/ppo.py --gym-id CartPole-v1
python cleanrl/c51.py --gym-id CartPole-v1
```

## PyBullet
```
poetry shell

poetry install -E pybullet
python cleanrl/td3_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/ddpg_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/sac_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
```

## Procgen 
```
poetry shell

poetry install -E procgen
python cleanrl/ppo_procgen.py --gym-id starpilot
python cleanrl/ppg_procgen.py --gym-id starpilot
```


## PPO + LSTM
```
poetry shell

poetry install -E atari
python cleanrl/ppo_atari_lstm.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/ppo_memory_env_lstm.py
```
