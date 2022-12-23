## Contributing to CleanRL

👍🎉 Thank you for taking the time to contribute! 🎉👍

Feel free to open an issue or a Pull Request if you have any questions or suggestions. You can also [join our Discord](https://discord.gg/D6RCjA6sVT) and ask questions there. If you plan to work on an issue, let us know in the issue thread to avoid duplicate work.

Good luck and have fun!

## Dev Setup

```bash
poetry install
poetry install --with atari
poetry install --with pybullet
```

Then you can run the scripts under the poetry environment in two ways: `poetry run` or `poetry shell`. 

* `poetry run`:
    By prefixing `poetry run`, your command will run in poetry's virtual environment. For example, try running
    ```bash
    poetry run python ppo.py
    ```
* `poetry shell`:
    First, activate the poetry's virtual environment by executing `poetry shell`. Then, the name of the poetry's
    virtual environment (e.g. `(cleanrl-ghSZGHE3-py3.9)`) should appear in the left side of your shell.
    Afterwards, you can directly run
    ```bash
    (cleanrl-ghSZGHE3-py3.9) python ppo.py
    ```


## Pre-commit utilities

We use [pre-commit](https://pre-commit.com/) to helps us automate a sequence of short tasks (called pre-commit "hooks") such as code formatting. In particular, we always use the following hooks when submitting code to the main repository.

* [**pyupgrade**](https://github.com/asottile/pyupgrade): pyupgrade upgrades syntax for newer versions of the language. 
* [**isort**](https://github.com/PyCQA/isort): isort sorts imported dependencies according to their type (e.g, standard library vs third-party library) and name.
* [**black**](https://black.readthedocs.io/en/stable/): black enforces an uniform code style across the codebase.
* [**autoflake**](https://github.com/PyCQA/autoflake): autoflake helps remove unused imports and variables.
* [**codespell**](https://github.com/codespell-project/codespell): codespell helps avoid common incorrect spelling.

You can run the following command to run the following hooks:

```bash
poetry run pre-commit run --all-files
```

which in most cases should automatically fix things as shown below: 

![](static/pre-commit.png)

## Contributing new algorithms

We welcome the contributions of new algorithms.

**Before opening a pull request**, please open an issue first to discuss with us since this is likely a sizable effort. Once we agree on the plan, feel free to make a PR to include the new algorithm.

To help ease the review process, here is a checklist template when contributing a new algorithm. See https://github.com/vwxyzjn/cleanrl/pull/137 as an example.

- [ ] I've read the [CONTRIBUTION](https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md) guide (**required**).
- [ ] I have ensured `pre-commit run --all-files` passes (**required**).
- [ ] I have contacted @vwxyzjn to obtain access to the [openrlbenchmark W&B team](https://wandb.ai/openrlbenchmark) (**required**).
- [ ] I have tracked applicable experiments in [openrlbenchmark/cleanrl](https://wandb.ai/openrlbenchmark/cleanrl) with `--capture-video` flag toggled on (**required**).
- [ ] I have updated the documentation and previewed the changes via `mkdocs serve`.
    - [ ] I have explained note-worthy implementation details.
    - [ ] I have explained the logged metrics.
    - [ ] I have added links to the original paper and related papers (if applicable).
    - [ ] I have added links to the PR related to the algorithm.
    - [ ] I have created a table comparing my results against those from reputable sources (i.e., the original paper or other reference implementation).
    - [ ] I have added the learning curves (in PNG format with `width=500` and `height=300`).
    - [ ] I have added links to the tracked experiments.
- [ ] I have updated the tests accordingly (if applicable).
