## Contributing to CleanRL

👍🎉 Thank you for taking the time to contribute! 🎉👍

Feel free to open an issue or a Pull Request if you have any questions or suggestions. You can also [join our Discord](https://discord.gg/D6RCjA6sVT) and ask questions there. If you plan to work on an issue, let us know in the issue thread to avoid duplicate work.

Good luck and have fun!

## Dev Setup

```bash
poetry install
poetry install -E atari
poetry install -E pybullet
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
    python ppo.py


## Code Formatting

We use [Pre-commit](https://pre-commit.com/) to sort dependencies, remove unused variables and imports, format code using black, and check word spelling. You can run the following command:

```bash
poetry run pre-commit run --all-files
```

## Contributing new algorithms

We welcome the contributions of new algorithms.

**Before opening a pull request**, please open an issue first to discuss with us since this is likely a sizable effort. Once we agree on the plan, feel free to make a PR to include the new algorithm.

To help ease the review process, here is a checklist:

1. **Code style**: Make sure you match the code style used in other implemented algorithms in CleanRL. In particular, `poetry run pre-commit run --all-files` will help auto-format the code.
2. **Empirical analysis and benchmark**: we adopt a similar guide from [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/CONTRIBUTING.md) with a bit of our spin. The implemented algorithm should come with **tracked** experiments that
    * match the reported performance in the paper (if applicable)
    * match the reported performance in a high-quality reference implementation (SB3, Tianshou, and others) (if applicable).
    * We should also add documentation on how exactly we want the tracked experiments to be done (i.e., what W&B project? should they capture video recording?)
3. **Documentation**: the proposed algorithm should also come with documentation at https://docs.cleanrl.dev/rl-algorithms/ to 
    * explain crucial implementation details
    * add links to the original paper and related papers (if applicable)
    * add links to the PR related to the algorithm
    * add links to the tracked experiments and benchmark results
4. **Tests**: the proposed algorithm should come with an end-to-end test (see examples [here](https://github.com/vwxyzjn/cleanrl/blob/master/tests/test_atari.py)) that ensures the algorithm does not crash. Other applicable tests are also welcome.


## Checklist:

Here is a checklist template when contributing a new algorithm. See https://github.com/vwxyzjn/cleanrl/pull/137 as an example.

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
