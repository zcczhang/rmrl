import copy
import random
import re
from typing import Optional, List, Union, Dict, Any, Literal, Type, Tuple

import gym
import numpy as np

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from allenact_plugins.stretch_manipulathor_plugin.dataset_utils import (
    TaskData,
    parse_obj_data_from_info,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_arm_environment import (
    StretchManipulaTHOREnvironment,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    STRETCH_ENV_ARGS,
    ExpRoomCombo,
    ExpRoomPickObjs,
    STATE_MEASURE_CONFIGS,
    REWARD_CONFIG,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_tasks.strech_pick_place import (
    StretchPickPlace,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_utils import position_distance
from allenact_plugins.utils import (
    load_json,
    dict_space_to_gym_space,
    bbox_intersect,
    states_measure,
    states_entropy,
    dict_space_to_gym_Box_space,
    MeasureBuffer,
)
from allenact_plugins.utils.config_utils import json_str

__all__ = [
    "StretchExpRoomPickPlaceTaskSampler",
    "StretchExpRoomPickPlaceResetFreeTaskSampler",
]


class StretchExpRoomPickPlaceTaskSampler(TaskSampler):
    """Task Sampler for Stretch Robot manipulation in ExpRoom.

    Args:
        mode: "train", "valid", "test" to determine the next task
        task_class: class to initialize task
        task_args: specific task kwargs to init task
        env_class: class to init env
        env_args: controller init args
        spec_args: specific env args to env (e.g. domain randomization)
        scene_directory: parent dir for scenes, empty if scene in `scenes` is abs path
        scenes: list of scene dataset, None for real-time sampling
        sensors: exp sensors
        max_step: task max step for each episode/phase
        action_space: task action space
        reward_config: task reward shaping config
        object_types: deprecated, use object_combos instead
        object_combos: specified pick-place objects combo in current sampler, (_, xyz) if random target
        max_tasks: max tasks to sample (for valid/test)
        seed: sampler seed
        deterministic_cudnn: deterministic cudnn for sampling
    """

    def __init__(
        self,
        *,
        mode: Literal["train", "valid", "test"] = "train",
        task_class: Type[StretchPickPlace] = StretchPickPlace,
        task_args: Optional[Dict[str, Any]] = None,
        env_class: Type[
            StretchManipulaTHOREnvironment
        ] = StretchManipulaTHOREnvironment,
        env_args: Dict[str, Any] = STRETCH_ENV_ARGS,
        spec_env_kwargs: Optional[dict] = None,
        scene_directory: str = "",
        scenes: Optional[List[str]] = None,
        sensors: Optional[List[Sensor]] = None,
        max_steps: int = 300,
        action_space: gym.Space = StretchPickPlace.action_space,
        rewards_config: Optional[Dict] = REWARD_CONFIG,
        object_types: Optional[List[str]] = None,
        object_combos: Optional[List[Tuple[str, str]]] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        deterministic_cudnn: bool = False,
        fixed_container: bool = False,
        fixed_container_pos_rot: Optional[Dict[str, dict]] = None,
        countertop_scale: Optional[Union[int, float, list, tuple]] = None,
        sample_kwargs: Optional[dict] = None,
        texture_randomization_keys: Optional[Union[str, Tuple[str]]] = None,
        # for recording
        record_during_training: bool = False,
        record_kwargs: Optional[dict] = None,
        process_ind: Optional[int] = None,
        **kwargs,
    ) -> None:
        if len(kwargs) > 0:
            get_logger().debug(
                f"{self.__class__.__name__} unused kwargs: {json_str(kwargs)}"
            )
        self.task_class = task_class
        self.task_args: Dict[str, Any] = task_args or dict(
            no_distractor=True, add_topdown_camera=False
        )
        if rewards_config is None:
            assert "reward_config" in self.task_args
        self.rewards_config = rewards_config
        self.environment_type = env_class
        self.env_args = env_args
        self.env: Optional[StretchManipulaTHOREnvironment] = None
        self.spec_env_kwargs = spec_env_kwargs
        self.sensors = sensors or []
        self.max_steps = max_steps
        self._action_space = action_space

        self.sampler_mode = mode
        self.max_tasks: Optional[int] = None
        self.num_tasks_generated = 0
        self.reset_tasks = max_tasks
        self.deterministic_sampling = deterministic_sampling
        self.task_seeds_list = task_seeds_list
        if task_seeds_list is not None:
            self.num_unique_seeds = len(self.task_seeds_list)
        else:
            self.num_unique_seeds = None
        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."
        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        self._last_env_seed: Optional[int] = None

        self._last_sampled_task: Optional[StretchPickPlace] = None
        self.set_seed(seed)
        if deterministic_cudnn:
            set_deterministic_cudnn()
        self.reset()

        # in case previous task sampler args
        if object_combos is None and object_types is not None:
            assert (isinstance(combo, tuple) for combo in object_types)
            object_combos = object_types
        if object_combos is None:
            assert scenes is not None, (
                "has to specify data to infer using object goal or random target goal "
                "when `object combos` is not specified"
            )
            object_combos = (
                [(x, "xyz") for x in ExpRoomPickObjs]
                if "xyz" in scenes[0]
                else ExpRoomCombo
            )
            get_logger().debug(f"object_combos is None, using {object_combos}")
        self.object_combos = object_combos

        self.sampled_combo_stats = {}  # num of sampled combos for debugging
        self.scenes = scenes
        # pick-container combo key to list of datapoints
        self.all_data_points: Dict[Tuple[str, str], List[dict]] = {}
        if scenes is not None:
            for scene in scenes:
                match = re.search(
                    r"tasks_obj_(\w+)_to_(\w+)_scene*.", scene.split("/")[-1]
                )
                assert match, scene
                # container could be "xyz" for random goals
                pick_obj, container = match.group(1), match.group(2)
                if (pick_obj, container) in object_combos:
                    self.sampled_combo_stats[(pick_obj, container)] = 0
                    data_point = load_json(scene_directory, scene)
                    pick_obj_id = data_point["init_object"]["object_id"]
                    container_id = (
                        data_point["goal_object"]["object_id"]
                        if container != "xyz"
                        else container
                    )
                    # NOTE: data key is unique obj id pairs, though in ExpRoom, obj names are already different
                    data_key = (pick_obj_id, container_id)
                    if data_key not in self.all_data_points.keys():
                        self.all_data_points[data_key] = []
                    self.all_data_points[data_key].append(data_point)
        else:
            # real-time sample in `next_task`
            for pick_obj, container in object_combos:
                self.sampled_combo_stats[(pick_obj, container)] = 0
                # NOTE: data key is obj name instead of id
                self.all_data_points[(pick_obj, container)] = []
        assert len(self.all_data_points) >= 1

        self.sample_info = dict(
            init_teleport_random_bounds=(0.2, 0.2),
            sample_countertop_offset=0.3,
            sample_limit=1000,
            objs_dist_thresh=0.3,
            objs_dist_upper_bound=np.inf,  # 0.5
            name2info=dict(),  # assume name is also unique in ExpRoom
        )  # cache unchanged info if sampling in real-time
        if sample_kwargs is not None:
            self.sample_info.update(sample_kwargs)
        self.scene_name = "FloorPlan_ExpRoom"  # specified for single scene
        self.prev_sampled_combo = None

        self.fixed_container = fixed_container
        if fixed_container_pos_rot is None:
            fixed_container_pos_rot = dict(
                position=dict(x=-0.5879999995231628, y=None, z=-0.45)
            )  # center of countertop
        self.fixed_container_pos_rot = fixed_container_pos_rot
        self.countertop_scale = countertop_scale
        self.texture_randomization_keys = texture_randomization_keys

        self.record_during_training = record_during_training
        self.process_ind = process_ind
        self.maybe_logging_point_cloud = record_during_training
        self.record_kwargs = record_kwargs or {}
        record_rule = self.record_kwargs.get("record_rule", [np.inf, 0, 0])
        self._record_start, self._record_interval, self._num_record_tasks = record_rule
        self._num_recorded = 0
        self.record = False

    def sample_new_combo(self, allow_repeat: bool = False) -> tuple:
        if self.prev_sampled_combo is not None and len(self.all_data_points) > 1:
            while True:
                # save combo to prevent sample the same from `last_sampled_task`
                chosen_obj_combo = random.choice(list(self.all_data_points.keys()))
                if allow_repeat or chosen_obj_combo != self.prev_sampled_combo:
                    break
        else:
            # initial combo
            chosen_obj_combo = random.choice(list(self.all_data_points.keys()))
        self.prev_sampled_combo = chosen_obj_combo
        return self.prev_sampled_combo

    def _process_obj_data_for_sampling(self, obj_name: str) -> dict:
        """sampling requires unchanged objectId, obj y in table, and obj size
        for bbox."""
        if obj_name not in self.sample_info["name2info"].keys():
            obj_info = self.env.get_object_by_name(obj_name)
            assert len(obj_info) == 1, f"{obj_name}, {obj_info}"
            obj_info = obj_info[0]
            obj_data_for_sample = dict(
                object_id=obj_info["objectId"],
                object_y=obj_info["position"]["y"],  # restrict to only reuse y
                object_size=obj_info["axisAlignedBoundingBox"]["size"],  # only use size
            )
            self.sample_info["name2info"][obj_name] = obj_data_for_sample
            return obj_data_for_sample
        else:
            return self.sample_info["name2info"][obj_name]

    def reset_seed(self):
        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                self._last_env_seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                self._last_env_seed = random.choice(self.task_seeds_list)
        else:
            self._last_env_seed = random.randint(0, 2 ** 31 - 1)
        set_seed(self._last_env_seed)

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[StretchPickPlace]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None
        if self.sampler_mode != "train" and self.length <= 0:
            return None
        if self.env is None:
            self.env = self._create_environment()

        self.reset_seed()

        chosen_obj_combo = self.sample_new_combo()
        # if not self.reset_free:
        if chosen_obj_combo in self.sampled_combo_stats:
            self.sampled_combo_stats[chosen_obj_combo] += 1
        else:
            # e.g. first random target goal
            self.sampled_combo_stats[chosen_obj_combo] = 1
        # in case random targets for RF
        data_lists = self.all_data_points.get(chosen_obj_combo, [])
        env_setup = False
        if len(data_lists) > 0:
            # TODO make it ordered for eval with dataset
            sample_data = random.choice(data_lists)
            visual_randomize_kwargs = sample_data.get("visual_randomize_kwargs", {})
            task_data = TaskData(**sample_data)
            self.reset_scene(
                self.scene_name, visual_randomize_kwargs=visual_randomize_kwargs
            )
        else:
            # real-time sampling
            pick_obj, container = chosen_obj_combo  # object names
            if not self.env.started:  # init reset to get default kwargs
                self.reset_scene(self.scene_name)
                init_agent = self.env.get_agent_location()
                teleport_state = {
                    "x": -0.6,
                    "y": init_agent["y"],
                    "z": -1.05,
                    "rotation": 270.0,
                    "horizon": init_agent["horizon"],
                    "standing": True,
                }
                # it would sample randomly agent init pos automatically then
                self.env.init_teleport_kwargs = teleport_state
                self.env.init_teleport_random_bounds = self.sample_info[
                    "init_teleport_random_bounds"
                ]

                countertop_id = self.env.get_object_by_name("ExpRoomTable")[0][
                    "objectId"
                ]
                if self.countertop_scale is not None:
                    self.env.scale_object(
                        object_id=countertop_id,
                        scale=self.countertop_scale,
                        add_every_reset=True,
                    )

                random_target_space = self.env.hand_reachable_space_on_countertop(
                    countertop_id,
                    fixed_axis="z",
                    offset=self.sample_info["sample_countertop_offset"],
                )
                self.sample_info["countertop_id"] = countertop_id
                self.sample_info["random_target_space"] = random_target_space
            # Assume the scene is always ExpRoom!
            countertop_id = self.sample_info["countertop_id"]
            random_target_space = self.sample_info["random_target_space"]
            is_obj_goal = chosen_obj_combo[1] != "xyz"

            init_obj_data = self._process_obj_data_for_sampling(pick_obj)
            init_obj_id = init_obj_data["object_id"]
            obj_size = init_obj_data["object_size"]
            if is_obj_goal:
                goal_data = self._process_obj_data_for_sampling(container)
                goal_id = goal_data["object_id"]
                goal_size = goal_data["object_size"]
            else:
                goal_data, goal_id = None, None  # dummy placeholders
                goal_size = obj_size

            initial_agent_pose, sample_goal_pose = None, None
            space = dict_space_to_gym_space(random_target_space)
            for i in range(self.sample_info["sample_limit"]):
                sampled_obj_pose = {k: float(v[0]) for k, v in space.sample().items()}
                sampled_obj_pose["y"] = init_obj_data["object_y"] + 0.01
                if (
                    is_obj_goal
                    and self.fixed_container
                    and self.fixed_container_pos_rot is not None
                ):
                    sample_goal_pose = self.fixed_container_pos_rot["position"]
                    if sample_goal_pose["y"] is None:
                        sample_goal_pose["y"] = goal_data["object_y"]
                else:
                    sample_goal_pose = {
                        k: float(v[0]) for k, v in space.sample().items()
                    }
                    if is_obj_goal:
                        sample_goal_pose["y"] = goal_data["object_y"] + 0.01
                sample_dist = position_distance(sampled_obj_pose, sample_goal_pose)
                if not bbox_intersect(
                    (sampled_obj_pose, obj_size), (sample_goal_pose, goal_size)
                ) and self.sample_info["objs_dist_thresh"] <= sample_dist <= (
                    self.sample_info["objs_dist_upper_bound"] or np.inf
                ):
                    # reset with randomness here
                    self.reset_scene(self.scene_name, visual_randomize_kwargs=None)

                    obj_ids = [init_obj_id]
                    if is_obj_goal:
                        obj_ids.append(goal_id)

                    countertop_info = self.env.get_object_by_id(countertop_id)
                    for obj_id in obj_ids:
                        if obj_id not in countertop_info["receptacleObjectIds"]:
                            get_logger().warning(
                                f"{obj_id} not on the table {countertop_id}, resampling"
                            )
                            continue

                    obj_infos = self.env.get_objects_by_ids(obj_ids, warning=True)
                    if obj_infos is None:
                        get_logger().debug("resampling as warning above")
                        continue
                    # using the real-time obj pose, again, in case edge cases
                    sampled_obj_pose["y"] = (
                        obj_infos[init_obj_id]["position"]["y"] + 0.01
                    )
                    if is_obj_goal:
                        sample_goal_pose["y"] = (
                            obj_infos[goal_id]["position"]["y"] + 0.01
                        )

                    initial_agent_pose = self.env.last_event.metadata["agent"]
                    event = self.env.teleport_object(
                        init_obj_id, sampled_obj_pose, fixed=False
                    )
                    if not event.metadata["lastActionSuccess"]:
                        get_logger().debug(
                            f"resampling as teleporting source obj {init_obj_id} failed..."
                        )
                        continue
                    if is_obj_goal:
                        event = self.env.teleport_object(
                            goal_id, sample_goal_pose, fixed=True
                        )
                        if not event.metadata["lastActionSuccess"]:
                            get_logger().debug(
                                f"resampling as teleporting goal obj {goal_id} failed..."
                            )
                            continue
                    break
                if i == self.sample_info["sample_limit"] - 1:
                    get_logger().warning(
                        f"reach the sampling limit {self.sample_info['sample_limit']}, "
                        f"using the last sample with obj, goal distance {sample_dist}"
                    )
            assert initial_agent_pose is not None
            self.env.wait()
            task_data_dict = dict(
                scene_name=self.scene_name,
                initial_agent_pos=initial_agent_pose,
                init_object=parse_obj_data_from_info(
                    obj_info=self.env.get_object_by_id(init_obj_id),
                    countertop_id=countertop_id,
                ),
                random_sample_space=random_target_space,
                is_obj_goal=is_obj_goal,
            )
            if is_obj_goal:
                goal_obj_info = parse_obj_data_from_info(
                    obj_info=self.env.get_object_by_id(goal_id),
                    countertop_id=countertop_id,
                )
                # update container pos rot after init sampling if no fixed pos rot specified
                if self.fixed_container and (
                    self.fixed_container_pos_rot is None
                    or len(self.fixed_container_pos_rot) != 2
                ):
                    self.fixed_container_pos_rot = dict(
                        position=goal_obj_info["object_location"],
                        rotation=goal_obj_info["object_rotation"],
                    )
                task_data_dict["goal_object"] = goal_obj_info
            else:
                task_data_dict["goal_object"] = dict(object_location=sample_goal_pose)
            task_data = TaskData(**task_data_dict)
            env_setup = True

        self.env.update_reset_stats()
        return self._create_task(task_data=task_data, setup_from_task=not env_setup)

    def _create_task(
        self, task_data: TaskData, setup_from_task: bool = False
    ) -> StretchPickPlace:
        cur_visual_randomization = self.env.current_visual_randomization or {}
        task_info = {
            # "id": "random%d" % random.randint(0, 2 ** 63 - 1)
            "id": task_data.task_name + f"{self._last_env_seed}",
            # for obj-goal combo embedding sensor
            "object_type": (
                task_data.init_object.object_name,
                task_data.goal_object.object_name,
            ),
            "sampled_combo_stats": {
                f"{k[0]}-{k[1]}": v for k, v in self.sampled_combo_stats.items()
            },
            "visual_randomize_kwargs": cur_visual_randomization,
        }
        init_task_kwargs = dict(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            reward_configs=self.rewards_config,
            task_data=task_data,
            record_during_training=self.record_during_training,
            process_ind=self.process_ind,
            # do env-wise check every step
            record_render_kwargs=dict(
                record_start=self._record_start, record_interval=self._record_interval,
            ),
        )
        init_task_kwargs.update(self.task_args)

        self.check_start_record_criteria()
        if self.record:
            # end recording in sensor
            init_task_kwargs["record_immediately"] = True
            debug_extra_views = self.record_kwargs.get("debug_extra_views", []) or []
            init_task_kwargs["record_render_kwargs"].update(
                mode=self.record_kwargs.get("mode", "debug"),
                debug_extra_views=debug_extra_views,
            )
            if "topdown" in debug_extra_views:
                init_task_kwargs["add_topdown_camera"] = True
            if "third-view" in debug_extra_views:
                init_task_kwargs["add_third_person_view_camera"] = True

        sampled_task = self.task_class(**init_task_kwargs)
        if setup_from_task:
            # TODO check=False when confident
            sampled_task.setup_from_task_data(check=True)

        self.num_tasks_generated += 1

        if self._last_sampled_task is None:
            # correct the num of reset here as maybe used reset to get initial positions
            self.env.num_resets = 1
            self.env.cum_steps_for_reset = 0

        self._last_sampled_task = sampled_task
        return self._last_sampled_task

    def check_start_record_criteria(self):
        """update whether (consecutively) recording."""
        if not self.record_during_training:
            return
        if self.env.cum_env_steps >= self._record_start and (
            self._last_sampled_task is None
            or self._last_sampled_task.record_should_start
        ):
            self.record = True
            get_logger().debug(f"record start at {self.env.cum_env_steps} steps")
        if self.record:
            self._num_recorded += 1
        if self._num_recorded > self._num_record_tasks:
            self._num_recorded = 0
            self.record = False

    @property
    def last_sampled_task(self) -> Optional[StretchPickPlace]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    def reset(self):
        self.max_tasks = self.reset_tasks
        self.num_tasks_generated = 0

    def set_seed(self, seed: int) -> None:
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    def _create_environment(self) -> StretchManipulaTHOREnvironment:
        init_env_kwargs = dict(env_args=self.env_args)
        init_env_kwargs.update(self.spec_env_kwargs or {})
        return self.environment_type(**init_env_kwargs)

    def reset_scene(
        self,
        scene_name: str = "FloorPlan_ExpRoom",
        visual_randomize_kwargs: Optional[dict] = None,
    ):
        if visual_randomize_kwargs is None:
            visual_randomize_kwargs = {}
        if self.texture_randomization_keys is not None:
            visual_randomize_kwargs[
                "texture_randomization_keys"
            ] = self.texture_randomization_keys
        self.env.reset(
            scene_name=scene_name,
            agentMode="stretch",
            visual_randomize_kwargs=visual_randomize_kwargs,
        )
        self.env.wait()

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        # return None if self.num_unique_seeds is None else self.num_unique_seeds
        return self.reset_tasks

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True


class StretchExpRoomPickPlaceResetFreeTaskSampler(StretchExpRoomPickPlaceTaskSampler):
    """object_combos/object_types should already be determined for RF/RM. Has
    to specify whether using random_targets or two-phase (forward-backward);
    and measurement-led or periodic intervention criteria.

    Args:
        random_targets: whether using random targets for RM-RL
        obj_goal_rate: rate for object goal v.s. random point goal if using random_targets
        two_phase: whether using two-phase FB-RL
        irr_measure: whether using measurement-led intervention
        irr_measure_method: measurement name if using measurement-led intervention
        num_steps_for_resets: interval for periodic resets, infinity by default for RF/RM
        multi_objects: whether using multiple objects as sequential RM-RL (currently not used)
        reset_if_obj_dropped: oracle intervention if the picking object is falling off the receptacle
    """

    def __init__(
        self,
        *,
        random_targets: bool = True,
        obj_goal_rate: float = 0.3,
        two_phase: bool = False,
        irr_measure: bool = True,
        irr_measure_method: Optional[str] = None,
        num_steps_for_resets: Optional[int] = 1e4,
        multi_objects: bool = False,
        reset_if_obj_dropped: bool = False,
        **kwargs,
    ):
        assert (
            int(random_targets) + int(two_phase) == 1
        ), f"only 1 of random targets and two phase should be True: {random_targets}, {two_phase}"
        super().__init__(**kwargs)
        if two_phase:
            self.obj_initial_pos = dict(x=-0.12, y=None, z=-0.5)
            self.current_phase = 0  # 0, 1 for forward-backward
        self.two_phase = two_phase
        self.random_targets = random_targets
        self.obj_goal_rate = obj_goal_rate
        self.irr_measure = irr_measure
        if irr_measure:
            assert (
                irr_measure_method is not None
                and irr_measure_method in STATE_MEASURE_CONFIGS
            )
            num_steps_for_resets = np.inf  # not necessary though
            self.num_tasks_irr = 0
            # TODO configurable
            self.irr_measure_method = irr_measure_method
            self.state_measure = STATE_MEASURE_CONFIGS[irr_measure_method]
            if "num_phases_irr_tolerance" in self.state_measure["thresholds"]:
                # mostly dispersion-based measure
                self.num_steps_for_measure = self.max_steps
                self.history_buffer: Optional[dict] = None
            elif "time_horizons" in self.state_measure:
                # mostly distance-based measure
                memory_steps = self.state_measure["time_horizons"]["memory"]
                measure_steps = self.state_measure["time_horizons"]["measure_steps"]
                self.num_steps_for_measure = memory_steps + measure_steps
                self.history_buffer = MeasureBuffer(
                    fixed_history_length=self.num_steps_for_measure
                )
            else:
                raise NotImplementedError(self.state_measure)

            self.task_args["track_history"] = True
        self.num_steps_for_resets = (
            num_steps_for_resets if num_steps_for_resets is not None else np.inf
        )
        self.reset_if_obj_dropped = reset_if_obj_dropped

        # aim to track if fixed container moved in the long horizon
        self.goal_objects: Dict[str, TaskData.Obj] = dict()

        # fixed objects goal, as data points may consist of xyz later
        self.objects_goal = list(self.all_data_points.keys())
        assert (
            "xyz" not in obj and "xyz" not in goal for obj, goal in self.objects_goal
        ), self.objects_goal
        # storing objects to keep once and update task args
        self._objects_stored = False

        self.multi_objects = multi_objects  # TODO

        self.hard_reset = False

    def sample_new_combo(self, allow_repeat: bool = False) -> tuple:
        combo = random.choice(self.objects_goal)
        if self.hard_reset:
            self.prev_sampled_combo = combo
            return combo
        if self.two_phase:
            combo = combo if self.current_phase == 0 else (combo[0], "xyz")
        else:
            assert self.random_targets
            if self.is_obj_inside_container() or random.random() > self.obj_goal_rate:
                # no consecutive object goal tasks if only 1 pick-place obj combo
                combo = (combo[0], "xyz")
        self.prev_sampled_combo = combo
        return combo

    def is_obj_inside_container(self) -> bool:
        """prevent the next goal is immediately success."""
        assert not self.multi_objects
        if self._last_sampled_task is None:
            return False
        return (
            len(
                self.env.get_object_by_id(self.task_args["distractors_to_keep"][-1])[
                    "receptacleObjectIds"
                ]
            )
            > 0
        )

    def need_hard_reset(self) -> bool:
        if self.last_sampled_task is None or self.env is None:
            # init sample for hard reset
            return True
        elif self.reset_if_obj_dropped and self.last_sampled_task.is_obj_off_table():
            return True
        elif self.irr_measure:
            # no `num_steps_for_reset` check by default
            return self.irr_measure_check()
        elif self.num_steps_for_resets <= self.env.cum_steps_for_reset:
            return True
        else:
            return False

    def irr_measure_check(self) -> bool:
        assert self.irr_measure and self._last_sampled_task is not None
        if self.last_sampled_task.success:
            self.num_tasks_irr = 0
            if isinstance(self.history_buffer, MeasureBuffer):
                self.history_buffer.clear_history()
            else:
                self.history_buffer = None
            return False
        # as task will be sampled for every phase,
        # state_history only stored for 1 phase every time
        prev_history: dict = self._last_sampled_task.state_history
        if (
            "hand_and_obj" in self.state_measure["metrics"]
            and "hand_and_obj" not in prev_history
        ):
            prev_history = dict(
                hand_and_obj=np.concatenate(
                    [np.array(v) for v in prev_history.values()], axis=1
                )
            )
        if isinstance(self.history_buffer, MeasureBuffer):
            self.history_buffer.extend_histories(prev_history)
        elif self.history_buffer is None:
            self.history_buffer = prev_history
        else:
            # not necessary though, as unsuccess trails will always have `max_steps`
            def extend_array_from_dict1_to_dict2(dict1: dict, dict2: dict) -> dict:
                for key, value in dict1.items():
                    assert key in dict2
                    dict2[key] = np.vstack([dict2[key], value])
                return dict2

            self.history_buffer = extend_array_from_dict1_to_dict2(
                prev_history, self.history_buffer
            )
        if self.irr_measure_method in ["entropy", "std", "entropy"]:
            history_length = len(list(self.history_buffer.values())[0])
            if history_length < self.num_steps_for_measure:
                return False
            irr = self.dispersion_measure_check(
                method=self.state_measure["metrics"]["name"],
                history=self.history_buffer,
                thresholds=self.state_measure["thresholds"],
            )
            # phase-wise checking so clear after checking
            self.history_buffer = None
        elif self.irr_measure_method in ["euclidean", "dtw"]:
            # reset history buffer in method
            irr = self.distance_measure_check()
        else:
            raise NotImplementedError(self.irr_measure_method)
        return irr

    def dispersion_measure_check(
        self, method: str, history: dict, thresholds: dict
    ) -> bool:
        if self._last_sampled_task.success:
            self.num_tasks_irr = 0
            return False
        irr = False
        _is_irr = []
        for key, key_history in history.items():
            if len(key_history) > 0:
                if method == "std":
                    k_measure = states_measure(
                        key_history, metrics="std", mean_xyz=True
                    )
                elif "entropy" in method:
                    grid_size = self.state_measure.get("grid_size", None)
                    entropy_fn = self.state_measure.get("entropy_fn", "scipy")
                    # TODO check which space should be
                    if entropy_fn == "scipy":
                        space = self._last_sampled_task.task_data.random_sample_space
                        space = dict_space_to_gym_Box_space(space.to_dict())
                    else:
                        space = None
                    k_measure = states_entropy(
                        key_history,
                        entropy_fn=entropy_fn,
                        grid_size=grid_size,
                        world_space=space,
                    )
                else:
                    raise NotImplementedError(method)
                # update logging metrics
                self.state_measure["metrics"][key] = k_measure
                # assume the measurement is positive
                _is_irr.append(0 <= k_measure < thresholds[key])
        # i.e. all pass threshold for irreversible
        last_task_irr = 0 < len(_is_irr) <= sum(_is_irr)
        if last_task_irr:
            self.num_tasks_irr += 1
        if self.num_tasks_irr >= thresholds["num_phases_irr_tolerance"]:
            irr = True
            self.num_tasks_irr = 0

        elif self.num_tasks_irr > 1 and not last_task_irr:
            # depends on whether using consecutive or cumulative
            # elif not last_task_irr:
            self.num_tasks_irr = 0

        # # debug frame before reset
        # if irr:
        #     frame = self._last_sampled_task.render(
        #         "debug", debug_extra_view="third-view"
        #     )
        #     imageio.imwrite(
        #         f"{f_mkdir('~/RESET_FINAL_FRAME')}/{self.env.cum_env_steps}-{self.process_ind}.png",
        #         frame,
        #     )
        return irr

    def distance_measure_check(self):
        if self.last_sampled_task is not None and self.last_sampled_task.success:
            # note: the metrics will not be updated then!
            self.history_buffer.clear_history()
            self.num_tasks_irr = 0
            return False
        metrics = self.history_buffer.distance_measure(
            measure_method=self.irr_measure_method,
            merge_histories="and" in self.state_measure["thresholds"],
            mem_steps=self.state_measure["time_horizons"]["memory"],
            measure_steps=self.state_measure["time_horizons"]["measure_steps"],
        )
        if metrics is None:
            return False

        is_irr = True
        for key, k_measure in metrics.items():
            self.state_measure["metrics"][key] = k_measure
            if not (0 <= k_measure <= self.state_measure["thresholds"][key]):
                is_irr = False
        if is_irr:
            self.num_tasks_irr += 1
        if (
            self.history_buffer.history_length
            >= self.history_buffer.fixed_history_length
        ):
            self.num_tasks_irr = 0
            self.history_buffer.clear_history()
        return is_irr

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[StretchPickPlace]:
        do_hard_reset = False
        if self.need_hard_reset():
            # is_first_reset = self._last_sampled_task is None
            do_hard_reset = True
            # hard reset would sample both obj and goal
            self.hard_reset = True
            next_task = super().next_task(force_advance_scene)
            if not self._objects_stored:
                goal_object = next_task.task_data.goal_object
                # fixed the container(s) info for this sampler
                self.goal_objects[goal_object.object_name] = goal_object
                assert next_task.is_obj_goal, next_task.task_data
                pick_obj_id, goal_obj_id = (
                    next_task.source_obj_id,
                    next_task.goal_obj_id,
                )
                self.task_args["no_distractor"] = False
                # always keep all picked & placed objs in scene, even for random targets
                # make the last idx as goal object to for reusable
                self.task_args["distractors_to_keep"] = self.task_args.get(
                    "distractors_to_keep", []
                ) or [] + [pick_obj_id, goal_obj_id]
                self._objects_stored = True
            self.hard_reset = False
            # if is_first_reset:
            #     return next_task

        assert self.env is not None, "env should already be initialized!"
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None
        if self.sampler_mode != "train" and self.length <= 0:
            return None

        if do_hard_reset:
            # to fairly sample random target
            self.sampled_combo_stats[self.prev_sampled_combo] -= 1
            self.num_tasks_generated -= 1
            if self.record:
                self._num_recorded -= 1

        self.prev_sampled_combo = self.sample_new_combo(allow_repeat=False)
        if self.prev_sampled_combo in self.sampled_combo_stats:
            self.sampled_combo_stats[self.prev_sampled_combo] += 1
        else:
            # e.g. first random target goal
            self.sampled_combo_stats[self.prev_sampled_combo] = 1
        pick_obj, container = self.prev_sampled_combo
        is_obj_goal = "xyz" not in container

        countertop_id = self.sample_info["countertop_id"]
        pick_object_info = parse_obj_data_from_info(
            obj_info=self.env.get_object_by_id(
                self.task_args["distractors_to_keep"][-2]
            ),
            countertop_id=countertop_id,
        )
        container_obj_info = parse_obj_data_from_info(
            obj_info=self.env.get_object_by_id(
                self.task_args["distractors_to_keep"][-1]
            ),
            countertop_id=countertop_id,
        )

        sample_goal_pose = None
        random_target_space = self.sample_info["random_target_space"]
        if self.two_phase:
            if self.current_phase == 1:
                sample_goal_pose = copy.deepcopy(self.obj_initial_pos)
                if sample_goal_pose["y"] is None:
                    sample_goal_pose["y"] = self._process_obj_data_for_sampling(
                        pick_obj
                    )["object_y"]
        elif not is_obj_goal:  # for random targets
            # sample random targets
            space = dict_space_to_gym_space(random_target_space)
            # pick_obj_pos = pick_object_info["object_location"]
            pick_obj_pos = pick_object_info["axisAlignedBoundingBox"]["center"]
            pick_obj_size = goal_size = pick_object_info["axisAlignedBoundingBox"][
                "size"
            ]
            container_center, container_size = (
                container_obj_info["axisAlignedBoundingBox"]["center"],
                container_obj_info["axisAlignedBoundingBox"]["size"],
            )
            for i in range(self.sample_info["sample_limit"]):
                sample_goal_pose = {k: float(v[0]) for k, v in space.sample().items()}
                sample_dist = position_distance(pick_obj_pos, sample_goal_pose)
                if (
                    sample_goal_pose["y"] >= container_center["y"]  # not too low
                    and not bbox_intersect(
                        (pick_obj_pos, pick_obj_size), (sample_goal_pose, goal_size)
                    )
                    and not bbox_intersect(
                        (container_center, container_size),
                        (sample_goal_pose, goal_size),
                    )
                    and self.sample_info["objs_dist_thresh"] <= sample_dist
                ):
                    break
                if i == self.sample_info["sample_limit"] - 1:
                    get_logger().warning(
                        f"reach the sampling limit {self.sample_info['sample_limit']}, "
                        f"using the last sample with obj, random target distance {sample_dist}. "
                        f"sample: {sample_goal_pose}, obj: {pick_obj_pos}, container: {container_center}"
                    )
                    # debug TODO del
                    # self._last_sampled_task.visualize_goal_point(goal=sample_goal_pose)
                    # frame = self._last_sampled_task.render(
                    #     "debug", debug_extra_view="third-view"
                    # )
                    # imageio.imwrite(
                    #     f"{f_mkdir('~/REJECT_SAMPLING')}/{self.env.cum_env_steps}-{self.process_ind}.png",
                    #     frame,
                    # )

            assert sample_goal_pose is not None
        initial_agent_pose = self.env.last_event.metadata["agent"]
        task_data_dict = dict(
            scene_name=self.scene_name,
            initial_agent_pos=initial_agent_pose,
            init_object=pick_object_info,
            random_sample_space=random_target_space,
            is_obj_goal=is_obj_goal,
        )
        if is_obj_goal:
            # did not change anything in reset free
            task_data_dict["goal_object"] = container_obj_info
        else:
            task_data_dict["goal_object"] = dict(object_location=sample_goal_pose)
        task_data = TaskData(**task_data_dict)
        next_task = self._create_task(task_data=task_data, setup_from_task=False)
        if self.irr_measure:
            # check whether positive in task instance
            next_task.state_measure_metrics = {
                "num_tasks_irr": self.num_tasks_irr,
                **self.state_measure["metrics"],
            }
        if self.reset_if_obj_dropped:
            next_task.terminate_if_obj_dropped = True

        if self.two_phase:
            # switch phase for next task
            self.current_phase = 1 - self.current_phase
        return next_task
