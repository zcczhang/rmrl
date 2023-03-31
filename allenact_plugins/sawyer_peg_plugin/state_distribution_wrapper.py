from typing import Any, Union

import gym
import numpy as np


class StateDistributionViz(gym.Wrapper):
    """Example of usage of Point Cloud visualization wrapper for Sawyer Peg."""

    def __init__(self, env: Union[str, Any], num_steps_recording: int = 1e5):
        if isinstance(env, str):
            env = gym.make(env)
        super().__init__(env)
        self.env = env
        self.num_accumulate_steps = 0
        self.num_steps_recording = num_steps_recording
        self.points = []
        self.traj_points = []
        self._init_peg = None
        self.log_point_cloud = False

        self.env.table_space = (
            gym.spaces.Box(
                high=np.array([0.45, 0.75, 0.0]), low=np.array([-0.45, 0.45, -0.027])
            )
            if self.env.small_table
            else gym.spaces.Box(
                high=np.array([0.65, 1.0, 0.0]), low=np.array([-0.65, 0.2, -0.027])
            )
        )

    def start_new_recording_point_cloud(self):
        self.log_point_cloud = False
        self.points = []

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        if self.num_accumulate_steps >= self.num_steps_recording:
            self.log_point_cloud = True
            self.num_accumulate_steps = 0
        else:
            self.log_point_cloud = False
        self._init_peg = self.env.obj_pos
        if isinstance(self._init_peg, np.ndarray):
            self._init_peg = self._init_peg.tolist()
        self.traj_points = [self._init_peg]
        return obs

    def step(self, action: np.ndarray) -> tuple:
        rtn = super().step(action)

        self.num_accumulate_steps += 1
        peg_pos = self.env.obj_pos
        if isinstance(peg_pos, np.ndarray):
            peg_pos = peg_pos.tolist()
        self.traj_points.append(peg_pos)
        if self.env.done:
            color = [0, 255, 0] if self.env.info["success"] else [255, 0, 0]
            for point in self.traj_points:
                self.points.append(point + color)
            self.points.append(self.env._target_pos.tolist() + [255, 255, 255])
        return rtn
