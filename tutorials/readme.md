## Requirements

* pyenv
* poetry

# Setup env

```
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
poetry install
```

# Auto-format

```bash
isort . --skip wandb
autoflake -r --exclude wandb --in-place --remove-unused-variables --remove-all-unused-imports .
black -l 127 --exclude wandb .
```