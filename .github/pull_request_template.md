## Description
<!--- Provide a general summary of your changes in here-->

## Types of changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Bug fix
- [ ] New feature
- [ ] New algorithm
- [ ] Documentation

## Checklist:
<!--- Go over all the following points, and put an `x` in all the boxes that apply. -->
<!--- If you're unsure about any of these, don't hesitate to ask. We're here to help! -->
- [ ] I've read the [CONTRIBUTION](https://github.com/vwxyzjn/cleanrl/blob/master/CONTRIBUTING.md) guide (**required**).
- [ ] I have ensured `pre-commit run --all-files` passes (**required**).
- [ ] I have updated the documentation and previewed the changes via `mkdocs serve`.
- [ ] I have updated the tests accordingly (if applicable).

If you are adding new algorithm variants or your change could result in performance difference, you may need to (re-)run tracked experiments. See https://github.com/vwxyzjn/cleanrl/pull/137 as an example PR. 
- [ ] I have contacted [vwxyzjn](https://github.com/vwxyzjn) to obtain access to the [openrlbenchmark W&B team](https://wandb.ai/openrlbenchmark) (**required**).
- [ ] I have tracked applicable experiments in [openrlbenchmark/cleanrl](https://wandb.ai/openrlbenchmark/cleanrl) with `--capture-video` flag toggled on (**required**).
- [ ] I have added additional documentation and previewed the changes via `mkdocs serve`.
    - [ ] I have explained note-worthy implementation details.
    - [ ] I have explained the logged metrics.
    - [ ] I have added links to the original paper and related papers (if applicable).
    - [ ] I have added links to the PR related to the algorithm variant.
    - [ ] I have created a table comparing my results against those from reputable sources (i.e., the original paper or other reference implementation).
    - [ ] I have added the learning curves (in PNG format).
    - [ ] I have added links to the tracked experiments.
    - [ ] I have updated the overview sections at the [docs](https://docs.cleanrl.dev/rl-algorithms/overview/) and the [repo](https://github.com/vwxyzjn/cleanrl#overview)
- [ ] I have updated the tests accordingly (if applicable).

