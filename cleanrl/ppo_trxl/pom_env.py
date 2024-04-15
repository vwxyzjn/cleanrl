import os
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from reprint import output

gym.register(
    id="ProofofMemory-v0",
    entry_point="pom_env:PoMEnv",
    max_episode_steps=16,
)


class PoMEnv(gym.Env):
    """
    Proof of Concept Memory Environment

    This environment is intended to assess whether the policy's memory is working.
    The environment is based on a one dimensional grid where the agent can move left or right.
    At both ends, a goal is spawned that is either punishing or rewarding.
    During the very first two steps, the agent gets to know which goal leads to a positive or negative reward.
    Afterwards, this information is hidden in the agent's observation.
    The last value of the agent's observation is its current position inside the environment.
    Optionally and to increase the difficulty of the task, the agent's position can be frozen until the goal information is hidden.
    To further challenge the agent, the step_size can be decreased.
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode="human"):
        self._freeze = True
        self._step_size = 0.2
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2
        self._op = None
        self.render_mode = render_mode
        glob = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Create an array with possible positions
        # Valid local positions are one tick away from 0.0 or between -0.4 and 0.4
        # Valid global positions are between -1 + step_size and 1 - step_size
        num_steps = int(0.4 / self._step_size)
        lower = min(-2.0 * self._step_size, -num_steps * self._step_size) if not glob else -1 + self._step_size
        upper = max(3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size) if not glob else 1
        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = list(map(lambda x: round(x, 2), self.possible_positions))  # fix floating point errors

    def step(self, action):
        action = action[0]
        reward = 0.0
        done = False

        if self._num_show_steps > self._step_count:
            self._position += self._step_size * (1 - self._freeze) if action == 1 else -self._step_size * (1 - self._freeze)
            self._position = np.round(self._position, 2)

            obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)

            if self._freeze:
                self._step_count += 1
                return obs, reward, done, False, {}

        else:
            self._position += self._step_size if action == 1 else -self._step_size
            self._position = np.round(self._position, 2)
            obs = np.asarray([0.0, self._position, 0.0], dtype=np.float32)  # mask out goal information

        # Determine reward and termination
        if self._position == -1.0:
            if self._goals[0] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        elif self._position == 1.0:
            if self._goals[1] == 1.0:
                reward += 1.0 + self._min_steps * self._time_penalty
            else:
                reward -= 1.0 + self._min_steps * self._time_penalty
            done = True
        else:
            reward -= self._time_penalty

        self._step_count += 1

        return obs, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._position = np.random.choice(self.possible_positions)
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        self._goals = goals[np.random.permutation(2)]
        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs, {}

    def render(self):
        if self._op is None:
            self.init_render = False
            self._op = output()
            self._op = self._op.warped_obj
            os.system("cls||clear")

            for _ in range(6):
                self._op.append("#")

        num_grids = 2 * int(1 / self._step_size) + 1
        agent_grid = int(num_grids / 2 + self._position / self._step_size) + 1
        self._op[1] = "######" * num_grids + "#"
        self._op[2] = "#     " * num_grids + "#"
        field = [*("#     " * agent_grid)[:-3], *"A  ", *("#     " * (num_grids - agent_grid)), "#"]
        if field[3] != "A":
            field[3] = "+" if self._goals[0] > 0 else "-"
        if field[-4] != "A":
            field[-4] = "+" if self._goals[1] > 0 else "-"
        self._op[3] = "".join(field)
        self._op[4] = "#     " * num_grids + "#"
        self._op[5] = "######" * num_grids + "#"

        self._op[6] = "Goals are shown: " + str(self._num_show_steps > self._step_count)

        time.sleep(1.0)

    def close(self):
        if self._op is not None:
            self._op.clear()
            self._op = None


if __name__ == "__main__":
    env = gym.make("ProofofMemory-v0")
    o, _ = env.reset()
    env.render()
    done = False
    while not done:
        o, r, done, _, _ = env.step(1)
        env.render()
    env.close()
