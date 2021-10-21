# Examples

To run training scripts in other games:
```
poetry shell

# classic control
python cleanrl/dqn.py --gym-id CartPole-v1
python cleanrl/ppo.py --gym-id CartPole-v1
python cleanrl/c51.py --gym-id CartPole-v1

# atari
poetry install -E atari
python cleanrl/dqn_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/c51_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/ppo_atari.py --gym-id BreakoutNoFrameskip-v4
python cleanrl/apex_dqn_atari.py --gym-id BreakoutNoFrameskip-v4

# pybullet
poetry install -E pybullet
python cleanrl/td3_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/ddpg_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0
python cleanrl/sac_continuous_action.py --gym-id MinitaurBulletDuckEnv-v0

# procgen
poetry install -E procgen
python cleanrl/ppo_procgen.py --gym-id starpilot
python cleanrl/ppo_procgen_impala_cnn.py --gym-id starpilot
python cleanrl/ppg_procgen.py --gym-id starpilot
python cleanrl/ppg_procgen_impala_cnn.py --gym-id starpilot
```

<script id="asciicast-443625" src="https://asciinema.org/a/443625.js" async></script>
