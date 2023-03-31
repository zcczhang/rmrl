import gzip
import json
import random
from typing import List, Optional, Union, Dict, Any, Type

import gym

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_seed, set_deterministic_cudnn
from allenact.utils.system import get_logger
from allenact_plugins.robothor_plugin.robothor_constants import *
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from allenact_plugins.utils import MeasureBuffer, states_measure, states_entropy


class ObjectNavDatasetTaskSampler(TaskSampler):
    def __init__(
        self,
        *,
        scenes: List[str],
        scene_directory: str,
        sensors: Optional[List[Sensor]] = None,
        max_steps: int = 300,  # 500
        env_class: Type[RoboThorEnvironment] = RoboThorEnvironment,
        env_args: Dict[str, Any] = OBJNAV_DEFAULT_ENV_ARGS,
        task_class: Type[ObjectNavTask] = ObjectNavTask,
        task_args: Optional[Dict[str, Any]] = None,
        action_space: gym.Space = ObjectNavTask.action_space,
        rewards_config: Dict = DEFAULT_REWARD_CONFIG,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        loop_dataset: bool = True,
        allow_flipping: bool = False,
        randomize_materials: bool = False,
        randomize_lightning: bool = False,
        num_task_per_scene: Optional[int] = None,
        process_ind: Optional[int] = None,
        # Reset Free/Minimize kwargs setting
        # only measurement or periodical
        reset_free: bool = False,
        measurement_lead_reset: Optional[bool] = None,
        measure_method: Optional[str] = None,
        num_steps_for_reset: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.task_args = task_args or {}
        self.process_ind = process_ind
        if reset_free and len(scenes) != 1:
            get_logger().warning(
                f"process {process_ind} has more than 1 scenes: {scenes},"
                f"probably not expected in reset free settings."
            )
        if reset_free:
            if measurement_lead_reset:
                assert measure_method is not None
                self.measure_method = measure_method
                self.num_tasks_irr = 0
                self.state_measure = STATE_MEASURE_CONFIGS[measure_method]
                if "num_phases_irr_tolerance" in self.state_measure["thresholds"]:
                    # mostly dispersion-based measure
                    self.num_steps_for_measure = max_steps
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
            else:
                assert num_steps_for_reset is not None
                self.num_steps_for_reset = num_steps_for_reset
            self.measurement_lead_reset = measurement_lead_reset

        self.reset_free = reset_free
        self._initial_reset = True
        self.scenes = scenes
        self.episodes = {
            scene: ObjectNavDatasetTaskSampler.load_dataset(
                scene, scene_directory + "/episodes", n=num_task_per_scene
            )
            for scene in scenes
        }

        # Only keep episodes containing desired objects
        if "object_types" in kwargs:
            self.episodes = {
                scene: [
                    ep for ep in episodes if ep["object_type"] in kwargs["object_types"]
                ]
                for scene, episodes in self.episodes.items()
            }
            self.episodes = {
                scene: episodes
                for scene, episodes in self.episodes.items()
                if len(episodes) > 0
            }
            self.scenes = [scene for scene in self.scenes if scene in self.episodes]

        self.env_class = env_class
        self.task_class = task_class

        self.object_types = [
            ep["object_type"] for scene in self.episodes for ep in self.episodes[scene]
        ]
        self.unique_object_types = list(
            set(kwargs.get("object_types", self.object_types))
        )
        assert len(self.unique_object_types) > 1, self.unique_object_types
        self.sample_object_type_stats = {k: 0 for k in self.unique_object_types}

        self.env: Optional[RoboThorEnvironment] = None
        self.sensors = sensors or []
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0
        self.randomize_materials = randomize_materials
        self.randomize_lightning = randomize_lightning

        self._last_sampled_task: Optional[ObjectNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> RoboThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(
        scene: str, base_directory: str, n: Optional[int] = None
    ) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        if n is not None:
            data = data[:n]
        return data

    @staticmethod
    def load_distance_cache_from_file(scene: str, base_directory: str) -> Dict:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ObjectNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    def need_hard_reset(self):
        if not self.reset_free or self._initial_reset or self.last_sampled_task is None:
            return True
        if self.measurement_lead_reset:
            return self.irr_measure_check()
        else:
            if self.env.cum_steps_for_reset >= self.num_steps_for_reset:
                return True
            return False

    def irr_measure_check(self) -> bool:
        assert self.measurement_lead_reset and self._last_sampled_task is not None
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
        if isinstance(self.history_buffer, MeasureBuffer):
            self.history_buffer.extend_histories(prev_history)
        elif self.history_buffer is None:
            self.history_buffer = prev_history

        if self.measure_method in ["entropy", "std", "entropy"]:
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
        elif self.measure_method in ["euclidean", "dtw"]:
            # reset history buffer in method
            irr = self.distance_measure_check()
        else:
            raise NotImplementedError(self.measure_method)
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
                        key_history,
                        metrics="std",
                        mean_xyz=True,
                        normalize_first=len(key_history[0]) > 3,
                    )
                elif "entropy" in method:
                    grid_size = self.state_measure.get("grid_size", None)
                    entropy_fn = self.state_measure.get("entropy_fn", "npeet")
                    k_measure = states_entropy(
                        key_history,
                        entropy_fn=entropy_fn,
                        grid_size=grid_size,
                        world_space=self.state_measure.get("space", None),
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
        elif not last_task_irr:
            self.num_tasks_irr = 0
        return irr

    def distance_measure_check(self):
        if self.last_sampled_task is not None and self.last_sampled_task.success:
            # note: the metrics will not be updated then!
            self.history_buffer.clear_history()
            self.num_tasks_irr = 0
            return False
        metrics = self.history_buffer.distance_measure(
            measure_method=self.measure_method,
            merge_histories=False,
            mem_steps=self.state_measure["time_horizons"]["memory"],
            measure_steps=self.state_measure["time_horizons"]["measure_steps"],
            normalize_first=False,
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

    def sample_episode_scene(self, need_hard_reset: bool = False):
        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0
        if need_hard_reset:
            scene = self.scenes[self.scene_index]
            episode = self.episodes[scene][self.episode_index]
        else:
            assert self._last_sampled_task is not None
            scene = self.env.scene_name
            last_object_type = self._last_sampled_task.target_object_type
            while True:
                object_type = random.choice(self.unique_object_types)
                if object_type != last_object_type:
                    break
            episode = dict(object_type=object_type)
        return episode, scene

    def next_task(
        self, force_advance_scene: bool = False, force_hard_reset: bool = False
    ) -> Optional[ObjectNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.env is None:
            self.env = self._create_environment()

        need_hard_reset = self.need_hard_reset() or force_hard_reset
        episode, scene = self.sample_episode_scene(need_hard_reset=need_hard_reset)
        if scene.replace("_physics", "") != self.env.scene_name.replace("_physics", ""):
            if not self._initial_reset and self.reset_free:
                raise Exception(
                    f"unexpected new scene introduced in this sampler: "
                    f'{scene.replace("_physics", "")} v.s. {self.env.scene_name.replace("_physics", "")}'
                )
            self.env.reset(scene_name=scene)
        else:
            self.env.reset_object_filter(render_image=False)

        # if reset free, this will be for initial frame
        self.env.set_object_filter(
            object_ids=[
                o["objectId"]
                for o in self.env.last_event.metadata["objects"]
                if o["objectType"] == episode["object_type"]
            ],
            render_image=not need_hard_reset,
        )

        were_materials_randomized = False
        were_lightning_randomized = False
        if self.randomize_materials:
            if (
                "Train" in scene
                or int(scene.replace("FloorPlan", "").replace("_physics", "")) % 100
                < 21
            ):
                were_materials_randomized = True
                self.env.controller.step(action="RandomizeMaterials")
        if self.randomize_lightning:
            self.env.controller.step(action="RandomizeLighting")
            were_lightning_randomized = True

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        task_info = {
            "scene": scene,
            "object_type": episode["object_type"],
            "materials_randomized": were_materials_randomized,
            "lightning_randomized": were_lightning_randomized,
            "mirrored": True
            if self.allow_flipping and random.random() > 0.5
            else False,
        }
        if need_hard_reset:
            teleport_success = self.env.teleport(
                pose=episode["initial_position"],
                rotation=episode["initial_orientation"],
                horizon=episode.get("initial_horizon", 0),
            )
            if not teleport_success:
                get_logger().debug(
                    f"teleport failed as {self.env.last_event.metadata['errorMessage']}"
                )
                return self.next_task(
                    force_advance_scene=force_advance_scene, force_hard_reset=True
                )
            self.env.update_reset_stats()

            task_info.update(
                {
                    "initial_position": episode["initial_position"],
                    "initial_orientation": episode["initial_orientation"],
                    "initial_horizon": episode.get("initial_horizon", 0),
                    # "distance_to_target": episode.get("shortest_path_length"),
                    # "path_to_target": episode.get("shortest_path"),
                    "id": episode["id"],
                }
            )
        else:
            agent_state = self.env.agent_state()
            task_info.update(
                {
                    "initial_position": {
                        k: v for k, v in agent_state.items() if k in ["x", "y", "z"]
                    },
                    "initial_orientation": agent_state["rotation"],
                    "initial_horizon": agent_state["horizon"],
                    # "distance_to_target": self.env.distance_to_object_type(
                    #     task_info["object_type"]
                    # ),
                    "id": f"{self.process_ind}_{self.episode_index}",
                }
            )

        self.sample_object_type_stats[episode["object_type"]] += 1
        self._last_sampled_task = self.task_class(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
            process_ind=self.process_ind,
            sample_object_type_stats=self.sample_object_type_stats,
            state_measure_metrics={
                "num_tasks_irr": self.num_tasks_irr,
                **self.state_measure["metrics"],
            }
            if self.reset_free and self.measurement_lead_reset
            else None,
            **self.task_args,
        )
        if self._initial_reset:
            self._initial_reset = False
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks
        self.sample_object_type_stats = {k: 0 for k in self.unique_object_types}

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)
