from copy import deepcopy

import gymnasium as gym
import numpy as np


# Only for gymnasium v1.0.0
class SameModelSyncVectorEnv(gym.vector.SyncVectorEnv):
    def step(self, actions):
        observations, infos = [], {}
        for i, action in enumerate(gym.vector.utils.iterate(self.action_space, actions)):
            (_env_obs, self._rewards[i], self._terminations[i], self._truncations[i], env_info,) = self.envs[
                i
            ].step(action)

            if self._terminations[i] or self._truncations[i]:
                infos = self._add_info(
                    infos,
                    {"final_obs": _env_obs, "final_info": env_info},
                    i,
                )
                _env_obs, env_info = self.envs[i].reset()
            observations.append(_env_obs)
            infos = self._add_info(infos, env_info, i)

        # Concatenate the observations
        self._observations = gym.vector.utils.concatenate(self.single_observation_space, observations, self._observations)
        return (
            deepcopy(self._observations) if self.copy else self._observations,
            np.copy(self._rewards),
            np.copy(self._terminations),
            np.copy(self._truncations),
            infos,
        )
