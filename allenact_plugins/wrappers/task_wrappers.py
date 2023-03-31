from typing import Any, Type
from typing import List, Dict, Optional, Union

import gym
import numpy as np

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor, SensorSuite
from allenact.base_abstractions.task import Task
from allenact_plugins.wrappers.env_wrappers import EnvWrapperType

__all__ = ["TaskWrapper", "Phase0TaskWrapper", "Phase1TaskWrapper"]


class TaskWrapper(Task[Type[EnvWrapperType]]):
    def __init__(
        self,
        env: Type[EnvWrapperType],
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        **kwargs,
    ):
        assert all(
            hasattr(env, attr) for attr in ["info", "max_path_length"]
        ), "need to specify env.max_path_length, and using env.info for extra metrics"

        max_steps = env.max_path_length

        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self._gym_done = False

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    def render(self, mode: str = "rgb_array", *args, **kwargs) -> np.ndarray:
        return self.env.render(mode)

    def get_observations(
        self, *args, gym_obs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Any:
        return self.sensor_suite.get_observations(
            env=self.env, task=self, gym_obs=gym_obs
        )

    def reached_terminal_state(self) -> bool:
        return self._gym_done

    def close(self) -> None:
        self.env.close()

    def metrics(self) -> Dict[str, Any]:
        metrics = {
            **super().metrics(),
            **{
                k: v for k, v in self.env.info.items() if not isinstance(v, (list, str))
            },
        }
        if hasattr(self.env.env, "num_resets"):
            # deprecated
            metrics["num_resets"] = self.env.env.num_resets
        if hasattr(self.env.env, "num_hard_resets"):
            metrics[
                f"num_resets/{self.task_info['process_ind']}"
            ] = self.env.env.num_hard_resets
        return metrics

    def _step(self, action: int) -> RLStepResult:
        gym_obs, reward, self._gym_done, info = self.env.step(action=action)

        return RLStepResult(
            observation=self.get_observations(gym_obs=gym_obs),
            reward=reward,
            done=self.is_done(),
            info=info,
        )


# ===================================
#     Shared Phases Task Wrapper
# ===================================


class Phase0TaskWrapper(TaskWrapper):
    def __init__(
        self,
        env: Type[EnvWrapperType],
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(env=env, sensors=sensors, task_info=task_info, **kwargs)
        self.env = env

    def metrics(self) -> Dict[str, Any]:
        metrics = {
            **{
                f"phase0/{k}": v
                for k, v in super().metrics().items()
                if k != "task_info"
            },
            f"task_info": super().metrics()["task_info"],
        }
        if "success" in self.env.info.keys():
            metrics["success"] = int(self.env.info["success"])
        if hasattr(self.env.env, "num_hard_resets"):
            metrics["num_hard_resets"] = self.env.env.num_hard_resets
            metrics[
                f"num_hard_resets/{self.task_info['process_ind']}"
            ] = self.env.env.num_hard_resets
        if hasattr(self.env.env, "num_phase_0"):
            metrics["num_phase_0"] = self.env.env.num_phase_0
        return metrics


class Phase1TaskWrapper(TaskWrapper):
    def __init__(
        self,
        env: Type[EnvWrapperType],
        sensors: Union[SensorSuite, List[Sensor]],
        task_info: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(env=env, sensors=sensors, task_info=task_info, **kwargs)

    def metrics(self) -> Dict[str, Any]:
        metrics = {
            **{
                f"phase1/{k}": v
                for k, v in super().metrics().items()
                if k != "task_info"
            },
            f"task_info": super().metrics()["task_info"],
            # f"success_overall": int(self.env.success_overall),
        }
        if "success" in self.env.info.keys():
            metrics["success"] = int(self.env.info["success"])
        if hasattr(self.env.env, "num_hard_resets"):
            metrics["num_hard_resets"] = self.env.env.num_hard_resets
            metrics[
                f"num_hard_resets/{self.task_info['process_ind']}"
            ] = self.env.env.num_hard_resets
        if hasattr(self.env.env, "num_phase_1"):
            metrics["num_phase_1"] = self.env.env.num_phase_1
        return metrics
