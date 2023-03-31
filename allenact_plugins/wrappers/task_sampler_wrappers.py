import random
from typing import List, Dict, Optional, Union, Callable
from typing import Type

from gym.utils import seeding

from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import set_seed
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger
from . import DictObsSensor, SingleRGBViewSensor
from .env_wrappers import EnvWrapperType, EnvWrapper, EnvRecorder
from .task_wrappers import TaskWrapper, Phase0TaskWrapper

__all__ = ["SinglePhaseTaskSampler", "AdvTaskSampler"]


class SinglePhaseTaskSampler(TaskSampler):
    """TaskSampler for single phase task.

    need to define `task_selector`, or pass, and preferably env_wrapper
    """

    def __init__(
        self,
        *,
        gym_env_type: str,
        env_wrapper: Type[EnvWrapperType] = EnvWrapper,
        sensors: Optional[Union[SensorSuite, List[Sensor]]] = None,
        task_selector: Callable[[str], Union[type, List[type]]] = None,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        sensor_selector: Optional[Callable[[str], type]] = None,
        repeat_failed_task_for_min_steps: int = 0,
        extra_task_kwargs: Optional[Dict] = None,
        seed: Optional[int] = None,
        process_ind: int,
        record_during_training: bool = False,
        record_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        get_logger().debug(
            f"Initialized TaskSampler: {self.__class__.__name__}, kwargs: {locals()}"
        )
        super().__init__()

        self.env_name = gym_env_type

        self.sensors: SensorSuite
        if sensors is None:
            # noinspection PyTypeChecker
            self.sensors = SensorSuite([sensor_selector(self.env_name)])
        else:
            self.sensors = (
                SensorSuite(sensors)
                if not isinstance(sensors, SensorSuite)
                else sensors
            )

        self.max_tasks = max_tasks
        self.num_unique_seeds = num_unique_seeds
        self.deterministic_sampling = deterministic_sampling
        self.repeat_failed_task_for_min_steps = repeat_failed_task_for_min_steps
        self.extra_task_kwargs = (
            extra_task_kwargs if extra_task_kwargs is not None else {}
        )

        self._last_env_seed: Optional[int] = None
        self._last_task: Optional[TaskWrapper] = None
        self._number_of_steps_taken_with_task_seed = 0

        assert (not deterministic_sampling) or repeat_failed_task_for_min_steps <= 0, (
            "If `deterministic_sampling` is True then we require"
            " `repeat_failed_task_for_min_steps <= 0`"
        )
        assert (self.num_unique_seeds is None) or (
            0 < self.num_unique_seeds
        ), "`num_unique_seeds` must be a positive integer."

        self.num_unique_seeds = num_unique_seeds
        self.task_seeds_list = task_seeds_list
        if self.task_seeds_list is not None:
            if self.num_unique_seeds is not None:
                assert self.num_unique_seeds == len(
                    self.task_seeds_list
                ), "`num_unique_seeds` must equal the length of `task_seeds_list` if both specified."
            self.num_unique_seeds = len(self.task_seeds_list)
        elif self.num_unique_seeds is not None:
            self.task_seeds_list = list(range(self.num_unique_seeds))
        if num_unique_seeds is not None and repeat_failed_task_for_min_steps > 0:
            raise NotImplementedError(
                "`repeat_failed_task_for_min_steps` must be <=0 if number"
                " of unique seeds is not None."
            )

        assert (not deterministic_sampling) or (
            self.num_unique_seeds is not None
        ), "Cannot use deterministic sampling when `num_unique_seeds` is `None`."

        if (not deterministic_sampling) and self.max_tasks:
            get_logger().warning(
                "`deterministic_sampling` is `False` but you have specified `max_tasks < inf`,"
                " this might be a mistake when running testing."
            )

        if seed is not None:
            self.set_seed(seed)
        else:
            self.np_seeded_random_gen, _ = seeding.np_random(
                random.randint(0, 2 ** 31 - 1)
            )
        self.process_ind = process_ind

        self.num_tasks_generated = 0
        self.task_type = task_selector(self.env_name)
        self.env_wrapper = env_wrapper

        has_rgb = (
            len(
                [
                    sensor
                    for sensor in self.sensors.sensors.values()
                    if isinstance(sensor, (DictObsSensor, SingleRGBViewSensor))
                ]
            )
            > 0
        ) or record_during_training
        gpu_id = kwargs.get("devices", [-1])
        if not isinstance(gpu_id, int):
            gpu_id = int(gpu_id[0])
        env = self.env_wrapper(self.env_name, gpu_render_id=gpu_id if has_rgb else None)

        self.env = (
            EnvRecorder(env=env, **record_kwargs) if record_during_training else env
        )

    @property
    def length(self) -> Union[int, float]:
        return (
            float("inf")
            if self.max_tasks is None
            else self.max_tasks - self.num_tasks_generated
        )

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return None if self.num_unique_seeds is None else self.num_unique_seeds

    def _repeating_update(self) -> bool:
        repeating = False
        if self.num_unique_seeds is not None:
            if self.deterministic_sampling:
                self._last_env_seed = self.task_seeds_list[
                    self.num_tasks_generated % len(self.task_seeds_list)
                ]
            else:
                self._last_env_seed = self.np_seeded_random_gen.choice(
                    self.task_seeds_list
                )
        else:
            if self._last_task is not None:
                self._number_of_steps_taken_with_task_seed += (
                    self._last_task.num_steps_taken()
                )

            if (
                self._last_env_seed is not None
                and self._number_of_steps_taken_with_task_seed
                < self.repeat_failed_task_for_min_steps
                and self._last_task.cumulative_reward == 0
            ):
                repeating = True
            else:
                self._number_of_steps_taken_with_task_seed = 0
                self._last_env_seed = self.np_seeded_random_gen.randint(0, 2 ** 31 - 1)
        return repeating

    def _task_reset(self, repeating: bool):
        if repeating and hasattr(self.env, "same_seed_reset"):
            self.env.same_seed_reset()
        else:
            set_seed(self._last_env_seed)
            self.env.seed(self._last_env_seed)
            self.env.saved_seed = self._last_env_seed
            self.env.reset()

    def _update_next_task(self):
        self.num_tasks_generated += 1
        task_info = {
            # "id": "random%d" % random.randint(0, 2 ** 63 - 1),
            "id": self._last_env_seed,
            "process_ind": self.process_ind,
        }

        # get_logger().warning(
        #     f"seed: {self._last_env_seed}, random_n: {random.randint(0, 2 ** 63 - 1)}"
        # )

        self._last_task = self.task_type(
            **dict(env=self.env, sensors=self.sensors, task_info=task_info),
            **self.extra_task_kwargs,
        )
        return self._last_task

    def next_task(self, force_advance_scene: bool = False) -> Optional[TaskWrapper]:
        if self.length <= 0:
            return None
        repeating = self._repeating_update()
        self._task_reset(repeating)
        task = self._update_next_task()
        return task

    def close(self) -> None:
        self.env.close()

    @property
    def all_observation_spaces_equal(self) -> bool:
        return True

    def reset(self) -> None:
        self.num_tasks_generated = 0
        # self.env.reset()

    def set_seed(self, seed: int) -> None:
        self.np_seeded_random_gen, _ = seeding.np_random(seed)
        if seed is not None:
            set_seed(seed)

    @property
    def last_sampled_task(self):
        return self._last_task


class AdvTaskSampler(SinglePhaseTaskSampler):
    """Task sampler for two phases tasks.

    In adversarial and/or reset free env, there should be attributes:
    `phase0_success`, `phase1_success` and `success_overall`. To log num
    of phases, there should be attributes: `num_phase_0` and
    `num_phase_0`
    """

    def __init__(
        self,
        gym_env_type: str,
        env_wrapper: Type[EnvWrapperType] = EnvWrapper,
        sensors: Optional[Union[SensorSuite, List[Sensor]]] = None,
        task_selector: Callable[[str], Union[type, List[type]]] = None,
        max_tasks: Optional[int] = None,
        num_unique_seeds: Optional[int] = None,
        task_seeds_list: Optional[List[int]] = None,
        deterministic_sampling: bool = False,
        sensor_selector: Optional[Callable[[str], type]] = None,
        repeat_failed_task_for_min_steps: int = 0,
        extra_task_kwargs: Optional[Dict] = None,
        seed: Optional[int] = None,
        mode: str = "train",
        **kwargs,
    ):
        super().__init__(**prepare_locals_for_super(locals()))
        self.mode = mode
        self.phase0_task, self.phase1_task = self.task_type

    def _update_next_task(self):
        self.num_tasks_generated += 1
        task_info = {
            "id": self._last_env_seed,
            "process_ind": self.process_ind,
        }

        # if repeating the task while the env has not been change out of a threshold
        repeat_task = hasattr(self.env.env, "repeat") and self.env.env.repeat

        if issubclass(type(self._last_task), Phase0TaskWrapper):
            # last task is phase 0 => next task is phase 1
            if repeat_task:
                new_task = self.phase0_task(
                    **dict(env=self.env, sensors=self.sensors, task_info=task_info),
                    **self.extra_task_kwargs,
                )
                self.env.env.phase = 0
                self.env.env.phase0_success = False
                if hasattr(self.env.env, "num_phase_0"):
                    self.env.env.num_phase_0 += 1
            else:
                new_task = self.phase1_task(
                    **dict(env=self.env, sensors=self.sensors, task_info=task_info),
                    **self.extra_task_kwargs,
                )
                self.env.env.phase = 1
                if hasattr(self.env.env, "num_phase_1"):
                    self.env.env.num_phase_1 += 1
        else:
            # last task is phase 1 or None (i.e. the first task) => next phase 0 task
            if self.mode != "train":
                # evaluate both phases, in the future, may only need eval forward phase
                self.env = self.env_wrapper(self.env_name)
            if repeat_task:
                new_task = self.phase1_task(
                    **dict(env=self.env, sensors=self.sensors, task_info=task_info),
                    **self.extra_task_kwargs,
                )
                self.env.env.phase = 1
                if hasattr(self.env.env, "num_phase_1"):
                    self.env.env.num_phase_1 += 1
            else:
                new_task = self.phase0_task(
                    **dict(env=self.env, sensors=self.sensors, task_info=task_info),
                    **self.extra_task_kwargs,
                )
                self.env.env.phase = 0
                self.env.env.phase0_success = False
                if hasattr(self.env.env, "num_phase_0"):
                    self.env.env.num_phase_0 += 1
        self._last_task = new_task
        return self._last_task
