from typing import Optional, Tuple, Any, Union, Sequence, Callable

import clip
import gym
import numpy as np
import torch

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor
from allenact_plugins.robothor_plugin.robothor_sensors import GPSCompassSensorRoboThor
from allenact_plugins.stretch_manipulathor_plugin.stretch_arm_environment import (
    StretchManipulaTHOREnvironment,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    PICKUP,
    RELEASE,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_tasks.strech_pick_place import (
    StretchPickPlace,
)

__all__ = [
    "RGBSensorStretch",
    "StretchPickPlaceStateObsSensor",
    "StretchPromptSensor",
    "StretchPolarSensor",
]


class RGBSensorStretch(RGBSensorThor):
    def __init__(
        self,
        rgb_keys: Union[str, Tuple[str, ...]] = ("wrist", "kinect"),
        concatenate_if_multiple: bool = True,
        *,
        use_resnet_normalization: bool = False,
        mean: Optional[Union[np.ndarray, Sequence[float]]] = None,
        stdev: Optional[Union[np.ndarray, Sequence[float]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        uuid: str = "rgb",
        output_shape: Optional[Tuple[int, ...]] = None,
        output_channels: int = 3,
        unnormalized_infimum: float = 0.0,
        unnormalized_supremum: float = 1.0,
        scale_first: bool = True,
        **kwargs: Any,
    ):
        self.rgb_keys = (rgb_keys,) if isinstance(rgb_keys, str) else rgb_keys
        self.concatenated = concatenate_if_multiple if len(rgb_keys) > 1 else False
        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(
        self, env: StretchManipulaTHOREnvironment, task: Optional[Task]
    ) -> np.ndarray:
        return env.render(rgb_keys=self.rgb_keys, cat=self.concatenated)


class StretchPickPlaceStateObsSensor(Sensor):
    """Position only observation of hand center (source obj) and target obj or
    random goal."""

    def __init__(
        self,
        uuid: str = "state_obs",
        *,
        state_obs_key: Union[str, Tuple[str, ...]] = ("hand", "target"),
        include_whether_obj_in_hand: bool = False,
    ):
        if isinstance(state_obs_key, str):
            state_obs_key = (state_obs_key,)
        assert (k in ("agent", "hand", "obj", "target") for k in state_obs_key)
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(state_obs_key) * 3 + int(include_whether_obj_in_hand),),
            dtype=np.float32,
        )
        self.state_obs_key = state_obs_key
        self.include_whether_obj_in_hand = include_whether_obj_in_hand
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchManipulaTHOREnvironment,
        task: StretchPickPlace,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        state_obs_dict = task.get_state_obs_dict()
        state_obs = np.concatenate(
            [
                [state_obs_dict[key]["position"][i] for i in ["x", "y", "z"]]
                for key in self.state_obs_key
            ],
        )
        if self.include_whether_obj_in_hand:
            obj_in_hand = float(
                env.is_object_at_low_level_hand(object_id=task.source_obj_id)
            )
            state_obs = np.concatenate([state_obs, [obj_in_hand],], dtype=np.float32)
        return state_obs


class StretchObjInHandSensor(Sensor):
    def __init__(
        self, uuid: str = "obj_in_hand",
    ):
        observation_space = gym.spaces.Discrete(1)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchManipulaTHOREnvironment,
        task: Optional[StretchPickPlace],
        *args,
        **kwargs,
    ) -> np.ndarray:
        # maybe directly infer from the oracle prompt
        obj_in_hand = env.is_object_at_low_level_hand(object_id=task.source_obj_id)
        return np.array([int(obj_in_hand)], dtype=np.int8)


class StretchActionAffordanceSensor(Sensor):
    def __init__(
        self, uuid: str = "action_affordance",
    ):
        observation_space = gym.spaces.Discrete(1)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: StretchManipulaTHOREnvironment,
        task: Optional[StretchPickPlace],
        *args,
        **kwargs,
    ) -> np.ndarray:
        if task.last_action_str in [PICKUP, RELEASE]:
            assert env.pick_release_correctly is not None
            good_action = env.pick_release_correctly
        elif task.last_action_success is not None:
            good_action = task.last_action_success
        else:
            assert task.num_steps_taken() == 0, task.num_steps_taken()
            good_action = True
        return np.array([int(good_action)], dtype=np.int8)


class StretchPromptSensor(Sensor):
    def __init__(
        self,
        uuid: str = "prompt",
        *,
        tokenizer: Callable = clip.tokenize,
        # is_point_goal: bool,
        # use_pick_phase: bool = False,
    ):
        self.tokenizer = tokenizer
        observation_space = self.make_observation_space()
        # self.is_point_goal = is_point_goal
        # self.use_pick_phase = use_pick_phase
        super().__init__(**prepare_locals_for_super(locals()))

    def make_observation_space(self):
        dummy_txt = "dummy"
        shape = self.tokenizer(dummy_txt).shape[-1]  # 77 for clip
        return gym.spaces.Box(low=0, high=49408, shape=(shape,), dtype=np.int32)

    def get_observation(
        self,
        env: StretchManipulaTHOREnvironment,
        task: Optional[StretchPickPlace],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        task_name = task.task_name
        return self.tokenizer(task_name).squeeze(0)


class StretchPolarSensor(GPSCompassSensorRoboThor):
    def __init__(
        self,
        uuid: str,
        *,
        state_obs_key: Union[str, Tuple[str, ...]] = ("hand", "target"),
        include_whether_obj_in_hand: bool = False,
    ):
        if isinstance(state_obs_key, str):
            state_obs_key = (state_obs_key,)
        assert (k in ("hand", "obj", "target") for k in state_obs_key)
        self.state_obs_key = state_obs_key
        self.include_whether_obj_in_hand = include_whether_obj_in_hand
        super().__init__(uuid=uuid)

    def _get_observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(
                2 * len(self.state_obs_key) + int(self.include_whether_obj_in_hand),
            ),
            dtype=np.float32,
        )

    def get_observation(
        self,
        env: StretchManipulaTHOREnvironment,
        task: Optional[StretchPickPlace],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        polar_obs = []
        if "polar_goal" in task.current_state_metadata:
            if "hand" in self.state_obs_key:
                polar_obs.append(task.current_state_metadata["polar_hand"])
            if "target" in self.state_obs_key:
                polar_obs.append(task.current_state_metadata["polar_goal"])
            if "obj" in self.state_obs_key:
                # assume obj is not accessible in agent obs!
                raise NotImplementedError
        else:
            # backup code, should be not used
            get_logger().warning("Deprecated polar obs calculations!")
            agent_state = env.get_agent_location()
            agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
            agent_rot_y = agent_state["rotation"]
            if isinstance(agent_rot_y, dict):
                agent_rot_y = agent_rot_y["y"]
            rotation_world_agent = self.quaternion_from_y_angle(agent_rot_y)

            if "hand" in self.state_obs_key:
                hand_state = env.get_absolute_hand_state()
                hand_pos = np.array(
                    [hand_state["position"][k] for k in ["x", "y", "z"]]
                )
                polar_hand = self._compute_pointgoal(
                    agent_position, rotation_world_agent, hand_pos
                )
                polar_obs.append(polar_hand)
            if "obj" in self.state_obs_key:
                obj_pos = env.get_object_by_id(task.task_data.init_object.object_id)[
                    "position"
                ]
                obj_position = np.array([obj_pos[k] for k in ["x", "y", "z"]])
                polar_obj = self._compute_pointgoal(
                    agent_position, rotation_world_agent, obj_position
                )
                polar_obs.append(polar_obj)
            if "target" in self.state_obs_key:
                # goal_pos = task.task_data.goal_object.object_location.to_dict()
                goal_pos = task.goal_pos
                goal_position = np.array([goal_pos[k] for k in ["x", "y", "z"]])
                polar_goal = self._compute_pointgoal(
                    agent_position, rotation_world_agent, goal_position
                )
                polar_obs.append(polar_goal)

        if self.include_whether_obj_in_hand:
            polar_obs.append(
                [float(env.is_object_at_low_level_hand(object_id=task.source_obj_id))]
            )
        return np.concatenate(polar_obs, dtype=np.float32)
