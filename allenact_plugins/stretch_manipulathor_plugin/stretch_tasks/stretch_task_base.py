import copy
import re
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Union

import gym
import numpy as np
from ai2thor.controller import Controller
from ai2thor.server import Event

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger
from allenact_plugins.stretch_manipulathor_plugin.dataset_utils import (
    TaskData,
    check_task_with_env,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_arm_environment import (
    StretchManipulaTHOREnvironment,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import REWARD_CONFIG
from allenact_plugins.stretch_manipulathor_plugin.stretch_utils import (
    position_distance,
    get_polar_pos,
)
from allenact_plugins.utils import (
    round_metrics,
    debug_texts_to_frame,
    add_boundary_from_success,
    assert_eq,
    dict_space_to_gym_space,
)

assert (
    gym.__version__ >= "0.20.0"
), f"{gym.__version__} < 0.20.0 which has the bug for seeding dict space"

__all__ = ["AbstractStretchTask"]


class AbstractStretchTask(Task[StretchManipulaTHOREnvironment]):
    """for each task there will be only 1 source and obj, but different tasks
    could have different source and target goal sampled by different
    samplers."""

    _actions = ()

    def __init__(
        self,
        env: StretchManipulaTHOREnvironment,
        sensors: List[Sensor],
        task_info: Optional[Dict[str, Any]] = None,
        max_steps: int = 200,
        *,
        # task spec kwargs
        reward_configs: dict = REWARD_CONFIG,
        success_threshold_obj: float = 0.1,
        # may different for obj or point goal
        success_threshold_random_targets: float = 0.05,
        no_distractor: bool = True,
        distractors_to_keep: Optional[List[str]] = None,
        add_topdown_camera: bool = False,
        add_third_person_view_camera: bool = False,
        task_data: Optional[Union[dict, TaskData]] = None,
        track_history: bool = False,
        terminate_if_obj_dropped: bool = False,
        # task prompt config
        use_polar_coord_goal: bool = True,
        use_pickup_phase: bool = True,
        # resample randomness to overwrite task data if not None
        sample_new_if_task_data: bool = False,
        record_during_training: bool = False,
        process_ind: Optional[int] = None,
        record_immediately: bool = False,
        record_render_kwargs: Optional[dict] = None,
        # debug_metrics_setting
        add_debug_metrics: bool = False,
        **kwargs,
    ) -> None:
        if kwargs is not None and len(kwargs) > 0:
            get_logger().debug(f"{self.__class__.__name__} unused kwargs: {kwargs}")
        super().__init__(
            env=env,
            sensors=sensors,
            task_info=task_info or {},
            max_steps=max_steps,
            **kwargs,
        )
        assert task_data is not None
        if isinstance(task_data, dict):
            self.task_data = TaskData(**task_data)  # prefer way to check the data
        else:
            assert isinstance(task_data, TaskData), type(task_data)
            self.task_data = task_data
        self.action_sequence_and_success = []
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_obj_to_goal_distance = None
        self.last_arm_to_obj_distance = None
        self.object_first_picked_up = False
        self.reward_configs = reward_configs
        self.reward_info = dict()
        # self.initial_object_metadata = self.env.get_current_object_locations()
        self._last_action_success = None
        self._last_action_str = None
        self._rewards: List[float] = []

        self.eplen_pickup = -1

        self.source_obj_id = self.task_data.init_object.object_id
        self.is_obj_goal = self.task_data.is_obj_goal
        self.goal_obj_id = None
        self.xyz_goal = None

        if self.is_obj_goal:
            self.goal_obj_id = self.task_data.goal_object.object_id
            # should be pre-sampled in dataset for repeatedly use
            # random_target_space = self.hand_reachable_space_on_countertop(
            #     self.task_data.init_object.countertop_id
            # )
        else:
            if sample_new_if_task_data:
                self.xyz_goal = self.sample_from_dict()
                with self.task_data.unlocked():
                    self.task_data.task_name = (
                        f"Put {self.task_data.init_object.object_name} "
                        f"to {round_metrics(tuple(self.xyz_goal.values()), n=4)}"
                    )
                    self.task_data.goal_object.object_location = TaskData.XYZ(
                        self.xyz_goal
                    )
            else:
                self.xyz_goal = (
                    self.task_data.goal_object.object_location
                )  # TaskData.XYZ
        self.success_threshold = (
            success_threshold_obj
            if self.is_obj_goal
            else success_threshold_random_targets
        )
        self.terminate_if_obj_dropped = terminate_if_obj_dropped

        # revert in add fn
        self._add_topdown_camera = None
        self._add_third_person_view_camera = None

        if no_distractor or distractors_to_keep is not None:
            # only keep `distractors_to_keep` and source and goal obj
            self.remove_objs_on_table(distractors_to_keep)
        if add_topdown_camera:
            self.add_topdown_camera()
        if add_third_person_view_camera:
            self.add_third_person_view_camera()

        self.track_history = track_history
        if track_history:
            self.state_history = dict(hand=[], obj=[])
        # set in task sampler
        self.state_measure_metrics = None

        self.use_polar_coord_goal = use_polar_coord_goal
        self.use_pickup_phase = use_pickup_phase

        self._step_state_dict = {}

        # dump debug info for metrics
        self.debug_info = {}

        # bool could be read from sensor
        self.record_during_training = record_during_training
        self.process_ind = process_ind
        self.record = record_immediately  # flag in sensor
        self.frame_histories = []
        self.record_render_kwargs = record_render_kwargs
        self.record_should_start = False
        self.add_debug_metrics = add_debug_metrics

        self.visualize_goal_point()

    @property
    def controller(self) -> Controller:
        return self.env.controller

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls) -> Tuple[str, ...]:
        return cls._actions

    @property
    def last_action_str(self) -> str:
        return self._last_action_str

    @property
    def last_action_success(self) -> str:
        return self._last_action_success

    def close(self) -> None:
        self.env.stop()

    @property
    def third_party_camera_frames_dict(self) -> dict:
        frames_dict = {}
        frames = self.env.last_event.third_party_camera_frames
        if self.env.added_third_party_camera_info is not None:
            if min(self.env.added_third_party_camera_info.values()) > 1:
                frames_dict["kinect"] = frames[0]
                frames_dict["wrist-centric"] = frames[1]
            for name, idx in self.env.added_third_party_camera_info.items():
                frames_dict[name] = frames[idx]
        if len(frames_dict) == 0 and frames is not None and len(frames) > 0:
            if len(frames) <= 2:
                frames_dict["kinect"] = frames[0]
                frames_dict["wrist-centric"] = frames[1]
            else:
                get_logger().warning(
                    f"more than one third party camera and not registered: "
                    f"{self.env.added_third_party_camera_info}"
                )
        return frames_dict

    def sample_from_dict(
        self,
        space: Optional[Union[gym.spaces.Dict, dict]] = None,
        # restrict to input a gym space, and possibly have seeded
        restrict_gym_space: bool = False,
    ) -> dict:
        if space is None:
            assert not restrict_gym_space
            assert self.task_data.random_sample_space.low is not None
            space = self.task_data.random_sample_space.to_dict()
        if not isinstance(space, gym.spaces.Dict):
            assert not restrict_gym_space
            space = dict_space_to_gym_space(space)
        return {k: float(v[0]) for k, v in space.sample().items()}

    @property
    def agent_location(self):
        return self.current_state_metadata["agent_location"]

    @property
    def source_obj_state(self):
        return self.current_state_metadata["source_obj_state"]

    @property
    def hand_state(self):
        return self.current_state_metadata["hand_state"]

    @property
    def goal_pos(self):
        return self.current_state_metadata["goal_pos"]

    @property
    def obj_goal_state(self):
        return self.current_state_metadata["obj_goal_state"]

    def get_point_goal(self) -> tuple:
        """in (rho, phi) formate."""
        if not self.use_polar_coord_goal:
            get_logger().warning(
                "`use_polar_coord_goal` is False but try to access the polar goal"
            )
        return self.current_state_metadata["polar_goal"]

    @property
    def current_state_metadata(self) -> dict:
        # update only once each step
        # not that current step starts from 0, and increment after `_step()`
        current_step = self.num_steps_taken()
        if current_step in self._step_state_dict:
            return self._step_state_dict[current_step]
        else:
            metadata = dict(
                agent_location=self.env.get_agent_location(),
                hand_state=self.env.get_absolute_hand_state(),
                source_obj_state=self.env.get_object_by_id(self.source_obj_id),
                obj_goal_state=self.env.get_object_by_id(self.goal_obj_id)
                if self.is_obj_goal
                else None,
            )
            goal_pos = (
                metadata["obj_goal_state"]["axisAlignedBoundingBox"]["center"]
                if self.is_obj_goal
                else self.xyz_goal.to_dict()
            )
            metadata["goal_pos"] = goal_pos
            if self.use_polar_coord_goal:
                agent_location = metadata["agent_location"]
                agent_rot_y = agent_location["rotation"]
                metadata["polar_goal"] = get_polar_pos(
                    euler_pos=goal_pos, base_pos=agent_location, y_angle=agent_rot_y
                )
                metadata["polar_hand"] = get_polar_pos(
                    euler_pos=metadata["hand_state"],
                    base_pos=agent_location,
                    y_angle=agent_rot_y,
                )

            self._step_state_dict = {current_step: metadata}

            if self.track_history:
                self.update_state_history()

            return metadata

    def update_state_history(self):
        if not self.track_history:
            return
        current_step = self.num_steps_taken()
        assert_eq(len(self.state_history["hand"]), current_step, reason="hand")
        assert_eq(len(self.state_history["obj"]), current_step, reason="obj")
        metadata = self.current_state_metadata
        hand_pos = metadata["hand_state"]["position"]
        self.state_history["hand"].append([hand_pos[i] for i in ["x", "y", "z"]])
        obj_pos = metadata["source_obj_state"]["position"]
        self.state_history["obj"].append([obj_pos[i] for i in ["x", "y", "z"]])

    @property
    def task_name(self) -> str:
        task_name = self.task_data.task_name
        if self.use_pickup_phase and not self.env.is_object_at_low_level_hand(
            self.source_obj_id
        ):
            pick_obj = self.task_data.init_object.object_name
            task_name = f"Pick {pick_obj.replace('_', ' ').lower()}."
        elif self.use_polar_coord_goal and not self.is_obj_goal:
            rho, phi = self.get_point_goal()
            task_name = re.sub(
                r"\((.*?),\s(.*?),\s(.*?)\)",
                f"({round(rho, 4)}, {round(phi, 4)})",
                task_name,
            )
        return task_name

    def get_state_obs_dict(self) -> OrderedDict:
        """could use to get xyz sensors, with all shared key `position`"""
        return OrderedDict(
            agent=self.env.last_event.metadata["agent"],
            hand=self.hand_state,
            obj=self.source_obj_state,
            target=dict(position=self.goal_pos),
        )

    def arm_distance_from_obj(self) -> float:
        return position_distance(self.source_obj_state, self.hand_state)

    def obj_distance_from_goal(self) -> float:
        return position_distance(self.source_obj_state, self.goal_pos)

    def body_distance_from_obj(self) -> float:
        # not used
        agent_state = dict(
            position={
                k: v for (k, v) in self.agent_location.items() if k in ["x", "y", "z"]
            }
        )
        return position_distance(self.source_obj_state, agent_state)

    def is_obj_off_table(self):
        return (
            self.current_state_metadata["source_obj_state"]["position"]["y"]
            < self.env.get_object_by_id(self.task_data.init_object.countertop_id)[
                "axisAlignedBoundingBox"
            ]["center"]["y"]
        )

    def get_original_object_distance(self) -> float:
        s_init = dict(position=self.task_data.init_object.object_location.to_dict())
        current_location = self.source_obj_state
        return position_distance(s_init, current_location)

    def _step(self, action: int) -> RLStepResult:
        raise NotImplementedError

    def _increment_num_steps_taken(self) -> None:
        super()._increment_num_steps_taken()
        self.env.increment_steps_stats()

    def success_criteria(self) -> bool:
        raise NotImplementedError

    @property
    def success(self):
        return self._success

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            get_logger().warning("THIS SHOULD NOT BE EXPECTED")
            return {}
        self.task_info.update(self.task_data.to_dict())

        result = super(AbstractStretchTask, self).metrics()

        if self.add_debug_metrics:
            result.update(self.calc_action_stat_metrics())

            final_obj_distance_from_goal = self.obj_distance_from_goal()
            result[
                "metric/average/final_obj_distance_from_goal"
            ] = final_obj_distance_from_goal
            final_arm_distance_from_obj = self.arm_distance_from_obj()
            result[
                "metric/average/final_arm_distance_from_obj"
            ] = final_arm_distance_from_obj
            final_obj_pickup = 1 if self.object_first_picked_up else 0
            result["metric/average/final_obj_pickup/total"] = final_obj_pickup

            original_distance = self.get_original_object_distance()
            result["metric/average/original_distance"] = original_distance

            pick_obj_name = self.task_data.init_object.object_name
            result[
                f"metric/average/final_obj_pickup/{pick_obj_name}"
            ] = self.object_first_picked_up
            if self.object_first_picked_up:
                destination_name = self.task_data.goal_object.object_name
                result[
                    f"metric/average/final_success/{destination_name}"
                ] = self._success

            if self.object_first_picked_up:
                ratio_distance_left = final_obj_distance_from_goal / (
                    original_distance + 1e-9
                )
                result["metric/average/ratio_distance_left"] = ratio_distance_left
                result["metric/average/eplen_pickup"] = self.eplen_pickup

            for sampled_combo, n in self.task_info["sampled_combo_stats"].items():
                result[f"sampled_combo/{sampled_combo}"] = n

            for k, v in self.debug_info.items():
                result[f"debug_metrics/{k}"] = v

        result["success"] = self._success
        result["final_object_pickup"] = int(self.object_first_picked_up)
        result["task_info"] = copy.deepcopy(result["task_info"])
        for k, v in self.reward_info.items():
            result[f"reward_info/{k}"] = v

        if self.state_measure_metrics is not None:
            metric_name = self.state_measure_metrics["name"]
            for k, v in self.state_measure_metrics.items():
                if k != "name" and v is not None:
                    result[f"state_measure/{k}_{metric_name}"] = v

        result["num_hard_resets"] = self.env.num_resets
        result[f"num_resets/{self.process_ind}"] = self.env.num_resets
        result["cum_steps_for_reset"] = self.env.cum_steps_for_reset
        result[f"cum_steps_for_reset/{self.process_ind}"] = self.env.cum_steps_for_reset

        return result

    def calc_action_stat_metrics(self) -> Dict[str, Any]:
        """for debugging."""
        action_stat = {
            "metric/action_stat/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat = {
            "metric/action_success/" + action_str: 0.0 for action_str in self._actions
        }
        action_success_stat["metric/action_success/total"] = 0.0

        seq_len = len(self.action_sequence_and_success)
        for (action_name, action_success) in self.action_sequence_and_success:
            action_stat["metric/action_stat/" + action_name] += 1.0
            action_success_stat[
                "metric/action_success/{}".format(action_name)
            ] += action_success
            action_success_stat["metric/action_success/total"] += action_success

        action_success_stat["metric/action_success/total"] /= seq_len

        for action_name in self._actions:
            action_success_stat[
                "metric/" + "action_success/{}".format(action_name)
            ] /= (action_stat["metric/action_stat/" + action_name] + 1e-7)
            action_stat["metric/action_stat/" + action_name] /= seq_len

        result = {**action_stat, **action_success_stat}
        return result

    def get_single_parent_receptacle_info(self, object_id: str) -> dict:
        obj_receptacle = self.env.get_parent_receptacles(object_id)
        if len(obj_receptacle) == 1:
            obj_receptacle_id = list(obj_receptacle)[0]
        else:
            raise ValueError(
                f"{object_id} has not exact one parent receptacle: {obj_receptacle}"
            )
        return self.env.get_object_by_id(obj_receptacle_id)

    def init_objects_receptacles(self) -> List[dict]:
        receptacles = [self.get_single_parent_receptacle_info(self.source_obj_id)]
        if self.is_obj_goal:
            obj_goal_receptacle = self.get_single_parent_receptacle_info(
                self.goal_obj_id
            )
            if obj_goal_receptacle["name"] != receptacles[0]["name"]:
                receptacles.append(obj_goal_receptacle)
        return receptacles

    def remove_objs_on_table(self, distractors_to_keep: Optional[List[str]] = None):
        goal_object_id = self.goal_obj_id if self.is_obj_goal else ""
        tables = (
            [self.env.get_object_by_id(self.task_data.init_object.countertop_id)]
            if self.task_data is not None
            else self.init_objects_receptacles()
        )
        objs_to_keep = (
            [self.source_obj_id, goal_object_id]
            if self.is_obj_goal
            else [self.source_obj_id]
        ) + (distractors_to_keep or [])
        for table in tables:
            objects_on_table = table["receptacleObjectIds"]
            for obj in objects_on_table:
                if obj not in objs_to_keep:
                    self.controller.step(dict(action="DisableObject", objectId=obj))
        self.env.wait()  # in case not stable

    def setup_from_task_data(self, check: bool = False):
        """make current self.env corresponding with states in the task data by
        teleport agent and object(s)"""
        task_data = self.task_data
        self.env.teleport_agent_to(
            **task_data.initial_agent_pos.to_dict(teleport_format=True)
        )
        self.source_obj_id = task_data.init_object.object_id
        self.env.teleport_object(
            task_data.init_object.object_id,
            task_data.init_object.object_location.to_dict(),
            fixed=False,
        )
        if task_data.is_obj_goal:
            self.is_obj_goal = True
            self.goal_obj_id = task_data.goal_object.object_id
            self.env.teleport_object(
                task_data.goal_object.object_id,
                task_data.goal_object.object_location.to_dict(),
                fixed=True,  # assume target obj fixed
            )
        else:
            self.is_obj_goal = False
            self.xyz_goal = self.task_data.goal_object.object_location
        self.env.wait()
        if check:
            self.check_task_data_with_env()

    def check_task_data_with_env(self):
        """check with the current env state and task data state."""
        task_data = self.task_data
        check_task_with_env(self.env, task_data)
        assert_eq(
            task_data.init_object.object_id, self.source_obj_id, reason="source obj id"
        )
        if task_data.is_obj_goal:
            assert self.is_obj_goal
            goal_obj_id = task_data.goal_object.object_id
            assert_eq(goal_obj_id, self.goal_obj_id, reason="goal obj id")
        else:
            assert not self.is_obj_goal
            assert_eq(
                self.xyz_goal.to_dict(),
                task_data.goal_object.object_location,
                reason="random target xyz",
            )

    def visualize_goal_point(self, goal: Optional[dict] = None) -> Event:
        if self.is_obj_goal:
            goal = dict(x=0, y=-10, z=0)
        else:
            goal = goal or self.goal_pos
        event = self.env.controller.step(dict(action="MoveGoalPoint", newPosition=goal))
        self.env.wait()
        return event

    def render(
        self,
        mode: str = "rgb",
        camera_name: str = "wrist-centric",
        debug_add_text: bool = False,
        debug_extra_views: Optional[Union[tuple, str]] = "third-view",
        add_success_boundary: bool = False,
        *args,
        **kwargs,
    ) -> np.ndarray:
        if mode in ["rgb", "rgb_array"]:
            if camera_name == "wrist-centric":
                return self.env.wrist_centric_frame
            if camera_name == "intel":
                return self.env.intel_frame
            elif camera_name == "kinect":
                return self.env.kinect_frame
            else:
                raise NotImplementedError(mode)
        elif mode == "debug":
            if "resolution" in kwargs:
                try:
                    self.env.change_resolution(
                        resolution=kwargs["resolution"], add_every_reset=True
                    )
                except AssertionError as e:
                    get_logger().warning(f"change resolution failed as {e}, passed")

            debug_frames = [
                debug_texts_to_frame(
                    # self.env.kinect_frame,
                    self.env.wrist_centric_frame,
                    self.debug_texts_dict(self.task_name),
                    **default_debug_render_kwargs(
                        self.env.resolution[0],
                        last_action_success=self.env.last_action_success,
                    ),
                )
                if debug_add_text
                else self.env.wrist_centric_frame
            ]

            if debug_extra_views is not None:
                if isinstance(debug_extra_views, str):
                    debug_extra_views = (debug_extra_views,)
                if "third-view" in debug_extra_views:
                    if (
                        not self._add_third_person_view_camera
                        or "third-view" not in self.env.added_third_party_camera_info
                    ):
                        get_logger().debug(
                            f"use third-person view third_party camera for debug mode"
                        )
                        self.add_third_person_view_camera()
                    frame = self.env.controller.last_event.third_party_camera_frames[
                        self.env.added_third_party_camera_info["third-view"]
                    ].copy()
                    if add_success_boundary:
                        frame = add_boundary_from_success(frame, success=self._success)
                    debug_frames.append(frame)
                if "topdown" in debug_extra_views:
                    if (
                        not self._add_topdown_camera
                        or "topdown" not in self.env.added_third_party_camera_info
                    ):
                        get_logger().debug(
                            f"use topdown third_party camera for debug mode"
                        )
                        self.add_topdown_camera()
                    frame = self.env.controller.last_event.third_party_camera_frames[
                        self.env.added_third_party_camera_info["topdown"]
                    ].copy()
                    if add_success_boundary:
                        frame = add_boundary_from_success(frame, success=self._success)
                    debug_frames.append(frame)
            # debug_frames = dict(
            #     wrist=debug_texts_to_frame(
            #         self.env.wrist_centric_frame,
            #         self.debug_texts_dict(self.task_data.task_name),
            #         **default_debug_render_kwargs(self.env.resolution[0]),
            #     )
            #     if debug_add_text
            #     else self.env.wrist_centric_frame,
            #     topdown=add_boundary_from_success(
            #         self.third_party_camera_frames_dict["topdown"].copy(), self._success
            #     ),
            # # )
            # frame = np.concatenate(list(debug_frames.values()), axis=1)
            frame = np.concatenate(debug_frames, axis=1)
            return frame
        else:
            raise NotImplementedError(mode)

    def step(self, action: Any) -> RLStepResult:
        rtn = super().step(action)
        if self.record:
            # cannot reuse the rendered img using rgb obs as rgb sensor used env.render()
            self.frame_histories.append(self.render(**self.record_render_kwargs or {}))
        if (
            self.record_render_kwargs is not None
            and self.env.cum_env_steps >= self.record_render_kwargs["record_start"]
            and (
                self.record_render_kwargs["record_interval"] == 0
                or self.env.cum_env_steps % self.record_render_kwargs["record_interval"]
                == 0
            )
        ):
            self.record_should_start = True
        return rtn

    def _check_logging(self):
        """sanity check for logging rewards and new task instance being sampled
        properly."""
        n_steps = self.num_steps_taken()
        # in case calling render before step incremental in super().step()
        assert len(self._rewards) <= n_steps + 1, f"{len(self._rewards)} > {n_steps}"

    def debug_texts_dict(self, task_str: str) -> dict:
        self._check_logging()
        texts_dict = dict(
            task=task_str,
            # steps=len(self._rewards),
            steps=self.num_steps_taken(),
            action=self._last_action_str or "Initialize",
            # hand_state=self.hand_state["position"],
            hand_pos=self.hand_state["position"],
            obj_pos=self.source_obj_state["position"],
            goal_pos=self.goal_pos,
            arm2obj=self.arm_distance_from_obj(),
            obj2target=self.obj_distance_from_goal(),
            first_grasped=self.object_first_picked_up,
            # maybe useful for debugging tasks
            # agent_location=round_metrics(self.env.get_agent_location()),
            # agent2target=round_metrics(self.body_distance_from_obj()),
        )
        if self.use_polar_coord_goal:
            texts_dict["polar_goal"] = self.current_state_metadata["polar_goal"]
            texts_dict["polar_hand"] = self.current_state_metadata["polar_hand"]
        if len(self._rewards) > 0:
            texts_dict.update(
                # step_reward=self._rewards[-1],
                **self.reward_info,
                cumulative_rewards=self.cumulative_reward,
                success=self._success,
            )
        return round_metrics(texts_dict)

    def add_topdown_camera(self):
        """note: third-party camera will be reset"""
        if (
            self._add_topdown_camera
            or "topdown" in self.env.added_third_party_camera_info
        ):
            self._add_topdown_camera = True
            return
        self._add_topdown_camera = True
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.env.add_third_party_camera(
            name="topdown",
            **event.metadata["actionReturn"],
            skyboxColor="white",
            # as task instances is supposed to sample every reset
            add_every_reset=False,
            update_if_existed=True,
        )

    def add_third_person_view_camera(self):
        if (
            self._add_third_person_view_camera
            or "third-view" in self.env.added_third_party_camera_info
        ):
            self._add_third_person_view_camera = True
            return
        self._add_third_person_view_camera = True
        if self.task_data.scene_name == "FloorPlan_ExpRoom":
            camera_config = dict(
                position=dict(x=-2, y=2.5035, z=2),
                rotation=dict(x=36.885, y=-215.31, z=0),
                fieldOfView=40,
            )
        else:
            camera_config = dict(
                position=dict(x=-1.5, y=2.5035, z=2.3),
                rotation=dict(x=36.885, y=-213.31, z=0),
                fieldOfView=45,
            )
        self.env.add_third_party_camera(
            name="third-view",
            **camera_config,
            skyboxColor="white",
            add_every_reset=False,
            update_if_existed=True,
        )


def default_debug_render_kwargs(height: int, last_action_success: bool = True) -> dict:
    color = [0, 255, 0] if last_action_success else [255, 0, 0]
    if 200 <= height <= 256:
        return dict(
            fontScale=0.3, org_x_init=5, org_y_init=10, org_y_increment=15, color=color
        )
    else:
        return dict(color=color)
