import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

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

    metadata = {"render_modes": ["human", "rgb_array", "debug_rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human"):
        self._freeze = True
        self._step_size = 0.2
        self._min_steps = int(1.0 / self._step_size) + 1
        self._time_penalty = 0.1
        self._num_show_steps = 2
        self.render_mode = render_mode
        glob = False

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Create an array with possible positions
        num_steps = int(0.4 / self._step_size)
        lower = min(-2.0 * self._step_size, -num_steps * self._step_size) if not glob else -1 + self._step_size
        upper = max(3.0 * self._step_size, self._step_size, (num_steps + 1) * self._step_size) if not glob else 1
        self.possible_positions = np.arange(lower, upper, self._step_size).clip(-1 + self._step_size, 1 - self._step_size)
        self.possible_positions = list(map(lambda x: round(x, 2), self.possible_positions))  # fix floating point errors

        # Pygame-related attributes for rendering
        self.window = None
        self.clock = None
        self.width = 400
        self.height = 80
        self.cell_width = self.width / (2 * int(1 / self._step_size) + 1)

    def step(self, action):
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
        self.rewards.append(reward)

        if done:
            info = {"reward": sum(self.rewards), "length": len(self.rewards)}
        else:
            info = {}

        self._step_count += 1

        return obs, reward, done, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rewards = []
        self._position = np.random.choice(self.possible_positions)
        self._step_count = 0
        goals = np.asarray([-1.0, 1.0])
        self._goals = goals[np.random.permutation(2)]
        obs = np.asarray([self._goals[0], self._position, self._goals[1]], dtype=np.float32)
        return obs, {}

    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            return

        # Initialize Pygame
        if not pygame.get_init():
            pygame.init()
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Proof of Memory Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create surface
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((255, 255, 255))  # Fill the background with white

        # Draw grid
        num_cells = 2 * int(1 / self._step_size) + 1
        for i in range(num_cells):
            x = i * self.cell_width
            pygame.draw.rect(canvas, (200, 200, 200), pygame.Rect(x, 0, self.cell_width, self.height), 1)

        # Draw agent
        agent_pos = int((self._position + 1) / self._step_size)
        agent_x = agent_pos * self.cell_width + self.cell_width / 2
        pygame.draw.circle(canvas, (0, 0, 255), (agent_x, self.height / 2), 15)

        # Draw goals
        show_goals = self._num_show_steps > self._step_count
        if show_goals:
            left_goal_color = (0, 255, 0) if self._goals[0] > 0 else (255, 0, 0)
            pygame.draw.rect(canvas, left_goal_color, pygame.Rect(0, 0, self.cell_width, self.height))
            right_goal_color = (0, 255, 0) if self._goals[1] > 0 else (255, 0, 0)
            pygame.draw.rect(
                canvas, right_goal_color, pygame.Rect(self.width - self.cell_width, 0, self.cell_width, self.height)
            )
        else:
            pygame.draw.rect(canvas, (200, 200, 200), pygame.Rect(0, 0, self.cell_width, self.height))
            pygame.draw.rect(
                canvas, (200, 200, 200), pygame.Rect(self.width - self.cell_width, 0, self.cell_width, self.height)
            )

        # Render text information
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Goals are shown: {show_goals}", True, (0, 0, 0))
        canvas.blit(text, (10, 10))

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode in ["rgb_array", "debug_rgb_array"]:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    env = PoMEnv(render_mode="human")
    o, _ = env.reset()
    img = env.render()
    done = False
    rewards = []
    const_action = 1
    while not done:
        o, r, done, _, _ = env.step(const_action)
        rewards.append(r)
        img = env.render()
    print(f"Total reward: {sum(rewards)}, Steps: {len(rewards)}")
    env.close()
