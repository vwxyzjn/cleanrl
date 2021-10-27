This folder contains experimental script for brax

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
git checkout -b refactor brax
poetry install
poetry install -E brax
poetry run pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
cd cleanrl/experiments
XLA_PYTHON_CLIENT_PREALLOCATE=false poetry run python ppo_brax_througput.py
```

Test throughput
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false poetry run  python brax_test.py
```