# Examples

## Atari
```
uv venv

uv pip install ".[atari]"
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/c51_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/ppo_atari.py --env-id BreakoutNoFrameskip-v4
python cleanrl/sac_atari.py --env-id BreakoutNoFrameskip-v4

# NEW: 3-4x side-effects free speed up with envpool's atari (only available to linux)
uv pip install ".[envpool]"
python cleanrl/ppo_atari_envpool.py --env-id BreakoutNoFrameskip-v4
# Learn Pong-v5 in ~5-10 mins
# Side effects such as lower sample efficiency might occur
uv run python ppo_atari_envpool.py --clip-coef=0.2 --num-envs=16 --num-minibatches=8 --num-steps=128 --update-epochs=3
```
### Demo

<script id="asciicast-443625" src="https://asciinema.org/a/443625.js" async></script>

You can also run training scripts in other games, such as:

## Classic Control
```
uv venv

python cleanrl/dqn.py --env-id CartPole-v1
python cleanrl/ppo.py --env-id CartPole-v1
python cleanrl/c51.py --env-id CartPole-v1
```

## Procgen 
```
uv venv

uv pip install ".[procgen]"
python cleanrl/ppo_procgen.py --env-id starpilot
python cleanrl/ppg_procgen.py --env-id starpilot
```


## PPO + LSTM
```
uv venv

uv pip install ".[atari]"
python cleanrl/ppo_atari_lstm.py --env-id BreakoutNoFrameskip-v4
```
