import math
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence, cast

import cv2
import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    END,
    LOOK_UP,
    LOOK_DOWN,
)
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.utils import spl_metric

__all__ = ["ObjectNavTask", "PointNavTask"]


class ObjectNavTask(Task[RoboThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, END, LOOK_UP, LOOK_DOWN)

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        *,
        # False for reset free
        done_always_terminate: bool = True,
        process_ind: Optional[int] = None,
        track_history: bool = False,
        state_measure_metrics: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.done_always_terminate = done_always_terminate
        self.process_ind = process_ind
        if track_history:
            self.state_history = {"agent": []}
        self.state_measure_metrics = state_measure_metrics
        self.track_history = track_history
        self._last_action_str = None

        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]

        self._all_metadata_available = env.all_metadata_available

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["taken_actions"] = []
        self.task_info["action_names"] = self.class_action_names()

        self._target_object_type = self.task_info["object_type"]
        if self._all_metadata_available:
            self.last_geodesic_distance = self.env.distance_to_object_type(
                self.target_object_type
            )
            self.optimal_distance = self.last_geodesic_distance
            self.closest_geo_distance = self.last_geodesic_distance

        self.last_expert_action: Optional[int] = None
        self.last_action_success = False

        self.debug_metrics = kwargs.get("sample_object_type_stats", {})

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    @property
    def target_object_type(self) -> str:
        return self._target_object_type

    @property
    def success(self):
        return self._success

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:
        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.class_action_names()[action]
        self._last_action_str = action_str

        if self.mirror:
            if action_str == ROTATE_RIGHT:
                action_str = ROTATE_LEFT
            elif action_str == ROTATE_LEFT:
                action_str = ROTATE_RIGHT

        self.task_info["taken_actions"].append(action_str)

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            if not self.done_always_terminate and not self._success:
                # keep the relatively fixed phase length in RF/RM training
                self._took_end_action = False
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1:
            self.travelled_distance += IThorEnvironment.position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        agent_state = self.env.last_event.metadata["agent"]
        if self.track_history:
            self.state_history["agent"].append(
                np.array(
                    [
                        agent_state["position"]["x"],
                        agent_state["position"]["z"],
                        # (action) scales are different even normalize
                        # agent_state["rotation"]["y"],
                        # agent_state["cameraHorizon"],
                    ]
                )
            )

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def _increment_num_steps_taken(self) -> None:
        super()._increment_num_steps_taken()
        self.env.increment_steps_stats()

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        if mode == "rgb":
            frame = self.env.current_frame.copy()
        elif mode == "depth":
            frame = self.env.current_depth.copy()
        elif mode == "debug":
            frame = self.env.current_frame.copy()
            if self.num_steps_taken() > 0:
                txts = dict(
                    n_step=len(self.task_info["taken_actions"]),
                    target_obj=self.task_info["object_type"],
                    action=self.task_info["taken_actions"][-1],
                    step_reward=np.round(self._rewards[-1], 4),
                    total_rewards=np.round(np.sum(self._rewards), 4),
                    geodesic_distance=np.round(self.last_geodesic_distance, 4),
                    success=self._success,
                )

                for i, txt in enumerate([f"{k}: {v}" for k, v in txts.items()]):
                    cv2.putText(
                        frame,
                        txt,
                        (10, 30 + i * 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )
        elif mode == "topdown":
            event = self.env.controller.step(action="GetMapViewCameraProperties")
            self.add_third_party_camera(
                "", **event.metadata["actionReturn"], skyboxColor="white",
            )
            frame = self.env.controller.last_event.third_party_camera_frames[-1]
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = frame[:, ::-1, :].copy()  # horizontal flip
            # print("mirrored render")
        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            o["objectType"] == self.task_info["object_type"]
            for o in self.env.visible_objects()
        )

    def add_third_party_camera(
        self,
        name: str,
        position: dict,
        rotation: dict,
        skyboxColor: str = "white",
        update_if_existed: bool = False,
        add_every_reset: bool = False,
        **kwargs,
    ):
        action_dict = dict(action="AddThirdPartyCamera")
        camera_kwargs = dict(
            position=position, rotation=rotation, skyboxColor=skyboxColor, **kwargs
        )
        action_dict.update(camera_kwargs)
        self.env.controller.step(**action_dict)

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.env.distance_to_object_type(
            self.task_info["object_type"]
        )

        # Ensuring the reward magnitude is not greater than the total distance moved
        max_reward_mag = 0.0
        if len(self.path) >= 2:
            p0, p1 = self.path[-2:]
            max_reward_mag = math.sqrt(
                (p0["x"] - p1["x"]) ** 2 + (p0["z"] - p1["z"]) ** 2
            )

        if self.reward_configs.get("positive_only_reward", False):
            if geodesic_distance > 0.5:
                rew = max(self.closest_geo_distance - geodesic_distance, 0)
        else:
            if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
            ):  # (robothor limits)
                rew += self.last_geodesic_distance - geodesic_distance

        self.last_geodesic_distance = geodesic_distance
        self.closest_geo_distance = min(self.closest_geo_distance, geodesic_distance)

        return (
            max(min(rew, max_reward_mag), -max_reward_mag)
            * self.reward_configs["shaping_weight"]
        )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        # if self._took_end_action:
        if self._last_action_str == END:
            if self._success:
                reward += self.reward_configs["goal_success_reward"]
            else:
                reward += self.reward_configs["failed_stop_reward"]
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_configs.get("reached_max_steps_reward", 0.0)

        self._rewards.append(float(reward))
        return float(reward)

    def get_observations(self, **kwargs) -> Any:
        obs = self.sensor_suite.get_observations(env=self.env, task=self)
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    if (
                        len(obs[o].shape) == 3
                    ):  # heuristic to determine this is a visual sensor
                        obs[o] = obs[o][:, ::-1, :].copy()  # horizontal flip
                    elif len(obs[o].shape) == 2:  # perhaps only two axes for depth?
                        obs[o] = obs[o][:, ::-1].copy()  # horizontal flip
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super(ObjectNavTask, self).metrics()
        if self._all_metadata_available:
            dist2tget = self.env.distance_to_object_type(self.task_info["object_type"])
            spl = spl_metric(
                success=self._success,
                optimal_distance=self.optimal_distance,
                travelled_distance=self.travelled_distance,
            )
            metrics = {
                **metrics,
                "success": self._success,
                "total_reward": np.sum(self._rewards),
                "dist_to_target": dist2tget,
                "spl": 0 if spl is None else spl,
                # **{
                #     f"debug_metrics/{k}-{self.env.scene_name}": v
                #     for k, v in self.debug_metrics.items()
                # },
                "initial_distance": self.optimal_distance,
                **self.debug_metrics,
            }
        metrics["num_hard_resets"] = self.env.num_resets
        metrics[f"num_resets/{self.process_ind}"] = self.env.num_resets
        metrics["cum_steps_for_reset"] = self.env.cum_steps_for_reset
        # metrics[
        #     f"cum_steps_for_reset/{self.process_ind}"
        # ] = self.env.cum_steps_for_reset

        if self.state_measure_metrics is not None:
            metric_name = self.state_measure_metrics["name"]
            for k, v in self.state_measure_metrics.items():
                if k != "name" and v is not None:
                    metrics[f"state_measure/{k}_{metric_name}"] = v

        return metrics

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True

        if end_action_only:
            return 0, False
        else:
            try:
                self.env.step(
                    {
                        "action": "ObjectNavExpertAction",
                        "objectType": self.task_info["object_type"],
                    }
                )
            except ValueError:
                raise RuntimeError(
                    "Attempting to use the action `ObjectNavExpertAction` which is not supported by your version of"
                    " AI2-THOR. The action `ObjectNavExpertAction` is experimental. In order"
                    " to enable this action, please install the (in development) version of AI2-THOR. Through pip"
                    " this can be done with the command"
                    " `pip install -e git+https://github.com/allenai/ai2thor.git@7d914cec13aae62298f5a6a816adb8ac6946c61f#egg=ai2thor`."
                )
            if self.env.last_action_success:
                expert_action: Optional[str] = self.env.last_event.metadata[
                    "actionReturn"
                ]
                if isinstance(expert_action, str):
                    if self.mirror:
                        if expert_action == "RotateLeft":
                            expert_action = "RotateRight"
                        elif expert_action == "RotateRight":
                            expert_action = "RotateLeft"

                    return self.class_action_names().index(expert_action), True
                else:
                    # This should have been caught by self._is_goal_in_range()...
                    return 0, False
            else:
                return 0, False


# just a dummy placeholder to accommodate import from AllenAct's code
PointNavTask = None
