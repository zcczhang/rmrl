from typing import Union, List

from allenact_plugins.wrappers import (
    TaskWrapper,
    Phase0TaskWrapper,
    Phase1TaskWrapper,
    SinglePhaseTaskSampler,
    AdvTaskSampler,
    MultiDiscretizeEnvWrapper,
)
from .sawyer_peg_env import SawyerPegV2Base

__all__ = [
    "SawyerPegMultiDiscretizeEnv",
    "MultiDiscretizeSawyerPegTaskSampler",
    "TwoPhasesSawyerPegMultiDiscreteTaskSampler",
]


class SawyerPegMultiDiscretizeEnv(MultiDiscretizeEnvWrapper):
    def __init__(self, env_name: str, num_bins: int = 7, **kwargs):
        super().__init__(env_name=env_name, num_bins=num_bins, **kwargs)


def task_selector(env_name: Union[str, SawyerPegV2Base]) -> Union[type, List[type]]:
    if isinstance(env_name, str):
        two_phases = (
            "RF" in env_name or "Adv" in env_name
        ) and "Random" not in env_name
    else:
        two_phases = env_name.two_phases and env_name.reset_free
    if two_phases:
        return [Phase0TaskWrapper, Phase1TaskWrapper]
    else:
        return TaskWrapper


class TwoPhasesSawyerPegMultiDiscreteTaskSampler(AdvTaskSampler):
    """Two adversarial RF Sawyer Peg w/ multi discrete action space."""

    def __init__(self, **kwargs):
        super().__init__(
            task_selector=task_selector,
            env_wrapper=SawyerPegMultiDiscretizeEnv,
            **kwargs
        )


class MultiDiscretizeSawyerPegTaskSampler(SinglePhaseTaskSampler):
    """Random Goal or Episodic Sawyer Peg."""

    def __init__(self, **kwargs):
        super().__init__(
            task_selector=task_selector,
            env_wrapper=SawyerPegMultiDiscretizeEnv,
            **kwargs
        )
