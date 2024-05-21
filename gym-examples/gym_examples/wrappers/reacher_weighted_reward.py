import gym


class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_dist_weight, reward_ctrl_weight):
        super().__init__(env)
        self.reward_dist_weight = reward_dist_weight
        self.reward_ctrl_weight = reward_ctrl_weight

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = (
            self.reward_dist_weight * info["reward_dist"]
            + self.reward_ctrl_weight * info["reward_ctrl"]
        )
        return obs, reward, terminated, truncated, info
