This repository is no longer maintained, as Gym is not longer maintained and all future maintenance of it will occur in the replacing [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) library. You can contribute Gymnasium examples to the Gymnasium repository and docs directly if you would like to. If you'd like to learn more about the transition from Gym to Gymnasium, you can read more about it [here](https://farama.org/Announcing-The-Farama-Foundation).

# Gym Examples
Some simple examples of Gym environments and wrappers.
For some explanations of these examples, see the [Gym documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).
