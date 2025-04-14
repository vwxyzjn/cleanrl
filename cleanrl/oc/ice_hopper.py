# File for IceWall hopper environment
# Original implementation: https://github.com/mklissa/gym-extensions/tree/master/gym_extensions/continuous/mujoco
# Based on modern gym hopper
# HopperIceWall = IceWallFactory(ModifiedHopperEnv)(ori_ind=-1, *args, **kwargs)
#   ModifiedHopperEnv is just hopper with kwargs["model_path"]
#   IceWallFactory(class_type) is a function returning IceWallEnv that inherits class_type
#   IceWallEnv(model_path, ori_ind, wall_height=.12, wall_pos_range=([2.8, 0.0], [2.8, 0.0]), n_bins=10, sensor_range=10., sensor_span=pi/2, *args, **kwargs)
#       Add walls to world body, first at random position in range. e.g., x (range[0][0], range[1][0])
#       Wall size = (0.25, 0.4, wall_height)
#       2 walls, 5 space between
#       _reset(): sample new wall position. Use it to modify model.geom_pos. Just 1 wall though...
#       _get_obs(): Need sensors for terrain in addition to full robot state obs
#           index_ratio = 2 indices per meter (each index is 0.5m long). terrain_read is 6D vector (zeros to start)
#           robot_xyz based on hopper foot
#           wall_length = wall_size[0] * 2
#           for each wall:
#               Get start and end x(wall_pos[0] -+ wall_size[0]).
#               diff = start_x - (robot_x - 1/index_ratio)  # Distance from robot to start of wall
#               if diff > 0: # Not there yet
#                   start_idx=round(diff*index_ratio), end_idx=start_idx + wall_length * index_ratio.
#               elif diff < 0 and diff >= -wall_length  # At wall, use end_diff as distance to end of wall
#                   start_idx=0, end_idx = round(end_diff)
#               else:  # Past wall, don't store in terrain_read
#               terrain_read[start_idx:end_idx] = 1.
#           Add terrain_read to obs
#       _is_in_collision(pos): Check (x,y) pos against wall pos (x,y) range. For MANUAL_COLLISION (default false)
#       _step(action): Either manual collision (-10 reward) or automatic (mujoco handles, no reward)


from os import path

import gym
import numpy as np
from dm_control import mjcf
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.hopper_v4 import HopperEnv
from gym.utils.ezpickle import EzPickle


class IceWall:
    height: float = 0.12
    x_len: float = 0.2
    y_len: float = 0.4

    def __init__(self, name, xy):
        self.mjcf_model = mjcf.RootElement(model=name)  # Root element of wall
        self.box = self.mjcf_model.worldbody.add("body", name="wall")  # body
        self.box.add(
            "geom",  # geom
            name="wall",
            type="box",
            pos=xy + [self.height / 2],
            size=[self.x_len, self.y_len, self.height],
            contype=1,
            conaffinity=1,
            rgba=[1.0, 0.0, 1.0, 1.0],
            density=0.00001,
        )


model_path = path.join(path.dirname(__file__), "assets", "hopper_icewall_1.xml")


class HopperIceBlock(HopperEnv):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        wall_x_range=(1.0, 2.8),
        wall_y_range=(0.0, 0.0),
        **kwargs,
    ):
        EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            wall_x_range,
            wall_y_range,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._wall_x_range = wall_x_range
        self._wall_y_range = wall_y_range
        self._wall_xy_range = tuple(zip(wall_x_range, wall_y_range))

        if exclude_current_positions_from_observation:
            observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        else:
            observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        MujocoEnv.__init__(self, model_path, 4, observation_space=observation_space, **kwargs)

    def reset_model(self):
        # Also reset randomly sample wall position
        wall_xy = self.np_random.uniform(low=self._wall_xy_range[0], high=self._wall_xy_range[1], size=2)
        self.model.geom("wall").pos[:-1] = wall_xy
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_obs(self):
        # Add 6D sensor reading vector to detect walls
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation
