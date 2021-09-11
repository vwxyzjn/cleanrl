```
cd cleanrl/atari
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
poetry install

isort . --skip wandb
autoflake -r --exclude wandb --in-place --remove-unused-variables --remove-all-unused-imports .
black -l 127 --exclude wandb .

pyenv shell $(cat .python-version)-prod

python dqn_atari_cpprb.py --track
```