# Modified mujoco environments with reversible forward direction
from os import path

import numpy as np
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv, MujocoEnv
from gym.utils.ezpickle import EzPickle


class HalfCheetahDirEnv(HalfCheetahEnv):
    # Simple change, just flip direction of movement reward
    def reverse(self):
        self._forward_reward_weight *= -1


class TMaze(MujocoEnv, EzPickle):
    # TODO: Complete
    # Two targets, sphere robot, remove most commonly-visited goal after some number of steps
    # https://github.com/anandkamat05/TDEOC/blob/master/baselines/Termination_DEOC/twod_tmaze.py
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }
    model_path = path.join(path.dirname(__file__), "assets", "tmaze.xml")

    def __init__(self, frame_skip: int = 4):
        EzPickle.__init__(self)
        super().__init__(self.model_path, frame_skip)
        self.target_count = np.array([0, 0])

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation
