from typing import Any, Type, Tuple
from typing import Dict, Optional, Union

import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from .env_wrappers import EnvWrapperType, EnvWrapper

__all__ = [
    "ObsSensor",
    "SingleRGBViewSensor",
    "DictObsSensor",
    "PhaseSensor",
    "IrrSensor",
]


class ObsSensor(Sensor[Type[EnvWrapperType], Task[Type[EnvWrapperType]]]):
    def __init__(
        self, uuid: str = "obs", num_stacked_frames: Optional[int] = 5, **kwargs: Any
    ):
        """
        Attributes:
            uuid : observation uuid
            num_stacked_frames : num of stacked frames in continuous space, None for not using
        """
        observation_space = self._get_observation_space(kwargs.get("env_name", None))
        super().__init__(**prepare_locals_for_super(locals()))

        self.num_stacked_frames = num_stacked_frames
        # followed by [o_{t}, o_{t-1}, o_{t-2}...]
        self.stacked_frames = None

    @staticmethod
    def _get_observation_space(env_name: Optional[str] = None) -> gym.Space:
        """need to implement based on different env if no `env_name`
        specified."""
        if env_name is not None:
            dummy_env = EnvWrapper(env_name=env_name, is_dummy_env=True)
            observation_space = dummy_env.observation_space
            dummy_env.close()
            return observation_space
        else:
            raise NotImplementedError

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if gym_obs is not None:
            current_frame = gym_obs
        else:
            current_frame = env.initial_observation

        if self.num_stacked_frames is None:
            # no frame stacking
            return np.array(current_frame, dtype="float32")

        if not self.stacked_frames:
            # repetitive initial stacked frames, instead of zero paddings
            self.stacked_frames = [current_frame] * self.num_stacked_frames
        self.stacked_frames = self.stacked_frames[:-1]
        self.stacked_frames.insert(0, current_frame)
        stacked_obs = np.array(self.stacked_frames, dtype=np.float32)
        if env.env.done and not hasattr(env.env, "reset_free"):
            # if reset version, then previous stacked frames should be discarded
            # for (adversarial) reset-free version, it is not necessary
            self.stacked_frames = None
        return stacked_obs


class SingleRGBViewSensor(ObsSensor):
    def __init__(
        self,
        uuid: str,
        view: str,
        image_size: Tuple[int, int] = (84, 84),
        num_stacked_frames: Optional[int] = None,
        **kwargs: Any,
    ):
        self._image_size = image_size
        self._view = view
        super().__init__(uuid=uuid, num_stacked_frames=num_stacked_frames, **kwargs)

    def _get_observation_space(self, env_name: Optional[str] = None) -> gym.Space:
        return gym.spaces.Box(
            0, 255, [3, *self._image_size], np.float32
        )  # channel first

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        current_frame = env.offscrean_render(
            view=self._view, resolution=self._image_size, transpose=True
        )
        current_frame = np.array(current_frame / 255.0, dtype=np.float32)

        if self.num_stacked_frames is None:
            return current_frame
        if not self.stacked_frames:
            self.stacked_frames = np.repeat(
                current_frame, self.num_stacked_frames, axis=0
            )
        else:
            self.stacked_frames = np.concatenate(
                (current_frame, self.stacked_frames[:-3, :, :]), axis=0
            )
        stacked_obs = np.array(self.stacked_frames, dtype=np.float32)
        if env.env.done and not hasattr(env.env, "reset_free"):
            # if reset version, then previous stacked frames should be discarded
            # for (adversarial) reset-free version, it is not necessary
            self.stacked_frames = None
        return stacked_obs


class DictObsSensor(ObsSensor):
    """Obs sensor with dict observation (rgb and/or state obs)"""

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Union[np.ndarray, Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Dict[str, Any]]:
        if gym_obs is not None:
            current_obs = gym_obs
        else:
            current_obs = env.initial_observation
        if isinstance(current_obs, np.ndarray):
            return super().get_observation(env, task, *args, current_obs, **kwargs)

        if self.num_stacked_frames is None:
            self.num_stacked_frames = 1

        if not self.stacked_frames:
            self.stacked_frames = dict()
            # repetitive initial stacked frames, instead of zero paddings
            for key, frame in current_obs.items():
                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                if frame.ndim == 3:
                    frame = np.array(frame / 255.0, dtype=np.float32)
                    # stacked in color channel
                    axis = 0 if frame.shape[0] == 3 else -1
                    self.stacked_frames[key] = np.repeat(
                        frame, self.num_stacked_frames, axis=axis
                    )
                else:
                    self.stacked_frames[key] = np.array(
                        [frame] * self.num_stacked_frames
                    )
        else:
            for key, frame in current_obs.items():
                if frame.ndim == 3:
                    frame = np.array(frame / 255.0, dtype=np.float32)
                    axis = 0 if frame.shape[0] == 3 else -1
                    self.stacked_frames[key] = np.concatenate(
                        (frame, self.stacked_frames[key][:-3, :, :]), axis=axis
                    )
                else:
                    self.stacked_frames[key] = np.concatenate(
                        ([frame], self.stacked_frames[key][:-1])
                    )
        stacked_obs = self.stacked_frames
        if env.env.done and not hasattr(env.env, "reset_free"):
            # if reset version, then previous stacked frames should be discarded
            # for (adversarial) reset-free version, it is not necessary
            self.stacked_frames = None
        return stacked_obs


class PhaseSensor(Sensor):
    """non-stacked phase sensor."""

    def __init__(self, uuid: str = "phase_sensor", phase_dim: int = 1, **kwargs: Any):
        observation_space = gym.spaces.Discrete(phase_dim)
        super().__init__(**prepare_locals_for_super(locals()))
        self.phase_dim = phase_dim

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        phase = env.env.phase if isinstance(env.env.phase, list) else [env.env.phase]
        assert len(phase) == self.phase_dim
        return np.array(phase)


class IrrSensor(Sensor):
    """irreversible label sensor."""

    def __init__(self, uuid: str = "irr_sensor", **kwargs: Any):
        observation_space = gym.spaces.Discrete(1)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return np.array([env.env.irreversible])
