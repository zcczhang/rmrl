from datetime import datetime
from typing import Optional, Union
from typing import TypeVar, Tuple

import cv2
import gym
import numpy as np

import allenact_plugins.utils as U

__all__ = ["EnvWrapper", "EnvWrapperType", "MultiDiscretizeEnvWrapper", "EnvRecorder"]

from allenact.utils.system import get_logger


class EnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env_name: Union[str, gym.Wrapper],
        gpu_render_id: Optional[int] = None,
        is_dummy_env: bool = False,
    ):
        env = gym.make(env_name) if isinstance(env_name, str) else env_name
        super().__init__(env)
        self._initial_observation: Optional[np.ndarray] = None
        if gpu_render_id is not None:
            self.env.setup_render_gpu_id(gpu_render_id)
        if not is_dummy_env:  # prevent generate img using GPU for dummy envs
            self.reset()  # generate initial observation

    def reset(self) -> np.ndarray:
        self._initial_observation = self.env.reset()
        return self._initial_observation

    @property
    def initial_observation(self) -> np.ndarray:
        assert (
            self._initial_observation is not None
        ), "Attempted to read initial_observation without calling reset()"
        res = self._initial_observation
        self._initial_observation = None
        return res

    def debug_img_no_text(self, resolution):
        corner2_img = self.sim.render(
            *resolution, mode="offscreen", camera_name="clearview"
        )
        try:
            corner3_img = self.sim.render(
                *resolution, mode="offscreen", camera_name="view_1"
            )
        except ValueError as e:
            get_logger().warning(f"incorrect camera, retry with corner2, error: {e}")
            corner3_img = self.sim.render(
                *resolution, mode="offscreen", camera_name="corner2"
            )
        cat_img = np.concatenate([corner2_img, corner3_img], axis=1)
        corner_img = self.sim.render(
            *resolution, mode="offscreen", camera_name="corner"
        )
        top_view_img = self.sim.render(
            *resolution, mode="offscreen", camera_name="topview"
        )
        return np.concatenate(
            [cat_img, np.concatenate([corner_img, top_view_img], axis=1)], axis=0
        )

    def render(
        self, mode="human", resolution=(320, 240), *args, **kwargs
    ) -> np.ndarray:
        """MuJoCo rendering base."""
        if mode == "human":
            return self.env.render(offscreen=False)
        if mode in ["rgb", "rgb_array"]:
            return self.env.sim.render(
                *resolution, mode="offscreen", camera_name="clearview"
            )
        elif mode == "debug":
            cat_img = self.debug_img_no_text(resolution)
            info = {
                "step": self.env.curr_path_length,
                **{k: v for k, v in self.info.items()},
            }
            if hasattr(self.env, "phase"):
                info["phase"] = self.env.phase
            for j, (k, v) in enumerate(info.items()):
                if k != "irreversible_rate":
                    v = round(v, 6) if isinstance(v, (float, int)) else str(v)
                    cv2.putText(
                        img=cat_img,
                        text=f"{k}: {v}",
                        org=(5, 15 * (j + 1)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 0, 0),
                    )
            return cat_img
        elif mode == "obs":
            return self.env.get_vis_obs(transpose=False)
        else:
            raise NotImplementedError


EnvWrapperType = TypeVar("EnvWrapperType", bound=EnvWrapper)


class MultiDiscretizeEnvWrapper(EnvWrapper):
    def __init__(
        self,
        env_name: str,
        num_bins: int = 7,
        gpu_render_id: Optional[int] = None,
        is_dummy_env: bool = False,
    ):
        super().__init__(
            env_name=env_name, gpu_render_id=gpu_render_id, is_dummy_env=is_dummy_env
        )
        low = self.action_space.low
        high = self.action_space.high
        num_actions = len(low)
        self.action_ranges = np.array(
            [np.linspace(low[i], high[i], num_bins) for i in range(num_actions)]
        )
        self.num_actions = num_actions
        self.action_space = gym.spaces.MultiDiscrete(
            [num_bins for _ in range(num_actions)]
        )

    def step(self, action: Union[np.ndarray, int]) -> tuple:
        continuous_action = self.action_ranges[np.arange(self.num_actions), action]
        rtn = super().step(continuous_action)
        self.info["action"] = action
        return rtn


class EnvRecorder(gym.Wrapper):
    def __init__(
        self,
        env: EnvWrapperType,
        mode: str = "debug",
        save: bool = False,
        save_path: str = ".",
        record_rule: Tuple[int, int, int] = (
            125000,
            125000,
            6,
        ),  # 125000 => 4M for 32 processes
        fps: int = 30,
        resolution: Tuple[int, int] = (320, 240),
    ):
        super().__init__(env=env)
        self.mode = mode
        self.save = save
        # generate file name for callback even if not saving locally to the folder
        self.save_path = U.f_join(
            save_path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"),
        )  # differentiate between different processes
        if save:
            U.f_mkdir(self.save_path)

        self.record_rule = record_rule
        self._record_start, self._record_interval, self._num_record_each = record_rule
        self.already_recorded_each = 0

        self.cum_env_steps = 0
        self.vid_writer = U.VideoTensorWriter(folder=self.save_path, fps=fps)
        self.record = False
        self._record_trigger = False
        self.resolution = tuple(resolution)

        self._render_fn = self.env.render
        self._step_fn = self.env.step
        for k, v in self.env.__dict__.items():
            setattr(self, k, v)

        self._initial_observation: Optional[np.ndarray] = None
        self.reset()  # generate initial observation

        # reset to None in TrainingVideoCallbackSensor after logging videos
        self.un_sync_video_file = None
        self.un_sync_video_frames = None

    @property
    def initial_observation(self) -> np.ndarray:
        assert (
            self._initial_observation is not None
        ), "Attempted to read initial_observation without calling reset()"
        res = self._initial_observation
        self._initial_observation = None
        return res

    def render(self, mode="human", **kwargs):
        return self._render_fn(mode=self.mode, resolution=self.resolution, **kwargs)

    def check_record_criteria(self):
        """call after super step."""
        if self.record:
            return
        extra_start_check = True
        if hasattr(self.env, "phase") and isinstance(self.env.phase, (float, int)):
            extra_start_check = self.env.phase == 0
        if (
            self._record_trigger
            and self.env.curr_path_length == 1
            and extra_start_check
        ):
            # the real start recording steps at the start of phase 0
            self.record = True
            self.vid_writer.clear()  # make sure indeed clear
            print(f"record started at {self.cum_env_steps} steps")
        elif (
            self.cum_env_steps > self._record_start
            and self.cum_env_steps % self._record_interval
            == 1  # as it is called at the end of each step
        ):
            # will start the recording at next phase 0
            self._record_trigger = True

    def step(self, action):
        self.cum_env_steps += 1
        obs, r, done, info = self._step_fn(action)
        self.check_record_criteria()
        if self.record:
            img = self.render(mode=self.mode)
            self.vid_writer.add_frame(img)
        return obs, r, done, info

    def reset(self, **kwargs):
        if self.record:
            self.already_recorded_each += 1
        if self.record and self.already_recorded_each >= self._num_record_each:
            self.un_sync_video_file = self.vid_writer.save(
                step=self.cum_env_steps, save=self.save
            )
            self.un_sync_video_frames = self.vid_writer.frames
            self.vid_writer.clear()
            self.already_recorded_each = 0
            self.record = False
            self._record_trigger = False
            print(f"record stopped with {self.cum_env_steps} steps")
        self._initial_observation = super().reset(**kwargs)
        return self._initial_observation
