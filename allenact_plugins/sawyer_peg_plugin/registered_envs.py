from typing import Union, Callable

import numpy as np
from omegaconf import ListConfig

from allenact.utils.system import get_logger
from .sawyer_peg_env import SawyerPegV2Base
from ..utils import register_gym_env

__all__ = ["SAWYER_PEG_ENVS", "SAWYER_PEG_QUICK_INIT_KWARGS", "parse_sawyer_peg_env"]

SAWYER_PEG_ENVS = dict(
    # low dim
    low_dim_obs_eval="SawyerPeg-v1",  # random position & hole
    low_dim_obs_in_domain_eval="SawyerPeg-v2",  # in domain valid
    low_dim_obs_train="RFSawyerPegRandomStdMeasure-v1",  # random, std measure
    # visual
    visual_eval="SawyerPegVisual-v1",  # random position & hole
    visual_in_domain_eval="SawyerPegVisual-v2",  # in domain valid
    visual_train="RFSawyerPegRandomStdMeasureVisual-v1",  # random, std measure
    # small table
    low_dim_small_table="SawyerPeg-v3",  # random position & hole, small table
    visual_small_table="SawyerPegVisual-v3",  # random position & hole, small table
    # state measure variations
    low_dim_std_measure="RFSawyerPegRandomStdMeasure-v1",
    visual_std_measure="RFSawyerPegRandomStdMeasureVisual-v1",
    low_dim_euclidean_measure="RFSawyerPegRandomEuclideanMeasure-v1",
    visual_euclidean_measure="RFSawyerPegRandomEuclideanMeasureVisual-v1",
    low_dim_dtw_measure="RFSawyerPegRandomDTWMeasure-v1",
    visual_dtw_measure="RFSawyerPegRandomDTWMeasureVisual-v1",
    low_dim_entropy_measure="RFSawyerPegRandomEntropyMeasure-v1",
    visual_entropy_measure="RFSawyerPegRandomEntropyMeasureVisual-v1",
)


# obs variations
LOW_DIM_KWARGS = dict(obs_keys="state", state_keys="all", compute_reward_kwargs="min")
VISUAL_OBS_KWARGS = dict(
    obs_keys=("rgb", "state"),
    state_keys=("tcp_center", "tcp_dist", "target"),
    compute_reward_kwargs="min",
)
# RF setting variations
RANDOM_TARGETS_KWARGS = dict(reset_free=True, random_targets=True, two_phases=False)
TWO_PHASES_KWARGS = dict(reset_free=True, random_targets=False, two_phases=True)
STATE_MEASURE_KWARGS = dict(almost_reversible=True, num_steps_for_hard_resets=np.inf)
FIXED_RESETS_KWARGS = dict(almost_reversible=False, num_steps_for_hard_resets=1e4)
# FIXED_RESETS_KWARGS = dict(almost_reversible=False, num_steps_for_hard_resets=2.5e4)
# generalization variations (initial pos is not super hard so not included)
FIXED_POS_HOLE_KWARGS = dict(random_init_box=False, random_box_hole=False)
RANDOM_POS_HOLE_KWARGS = dict(random_init_box=True, random_box_hole=True)


SAWYER_PEG_QUICK_INIT_KWARGS = dict(
    low_dim_kwargs=LOW_DIM_KWARGS,
    visual_obs_kwargs=VISUAL_OBS_KWARGS,
    random_targets_kwargs=RANDOM_TARGETS_KWARGS,
    two_phases_kwargs=TWO_PHASES_KWARGS,
    state_measure_kwargs=STATE_MEASURE_KWARGS,
    fixed_resets_kwargs=FIXED_RESETS_KWARGS,
    fixed_pos_hole_kwargs=FIXED_POS_HOLE_KWARGS,
    random_pos_hole_kwargs=RANDOM_POS_HOLE_KWARGS,
)


def parse_sawyer_peg_env(
    env_spec: Union[str, list, ListConfig, Callable], **kwargs
) -> Union[str, SawyerPegV2Base]:
    if not isinstance(env_spec, str):
        if isinstance(env_spec, (list, ListConfig)):
            assert (k.lower() in SAWYER_PEG_QUICK_INIT_KWARGS for k in env_spec)
            init_kwargs = {
                k: v
                for spec in env_spec
                for k, v in SAWYER_PEG_QUICK_INIT_KWARGS[spec.lower()].items()
            }
            get_logger().debug(
                f"initialize SawyerPeg env with quick kwargs {env_spec}, full kwargs: {init_kwargs}"
                + f", {kwargs}" * (len(kwargs) > 0)
            )
            return SawyerPegV2Base(**init_kwargs, **kwargs)
        else:
            assert not isinstance(
                env_spec, Callable
            ), f"{env_spec} has to be initialized with args"
            get_logger().debug(f"env object already initialized: {env_spec}")
            return env_spec
    if env_spec in SAWYER_PEG_ENVS:
        env = SAWYER_PEG_ENVS[env_spec]
        get_logger().debug(f"initialize env: {env_spec} for {env}")
        return env
    else:
        assert env_spec in SAWYER_PEG_ENVS.values()
        get_logger().debug(f"initialize env: {env_spec}")
        return env_spec


# Episodic evaluation env
# no visual
@register_gym_env("SawyerPeg-v1")
class SawyerPegV2Eval(SawyerPegV2Base):
    """episodic, random box and hole."""

    def __init__(self):
        super().__init__(reset_free=False, **RANDOM_POS_HOLE_KWARGS, **LOW_DIM_KWARGS)


@register_gym_env("SawyerPeg-v2")
class SawyerPegV2EvalInDomain(SawyerPegV2Base):
    """episodic, fixed box and hole."""

    def __init__(self):
        super().__init__(reset_free=False, **FIXED_POS_HOLE_KWARGS, **LOW_DIM_KWARGS)


# visual
@register_gym_env("SawyerPegVisual-v1")
class SawyerPegV2EvalVisual(SawyerPegV2Base):
    """episodic, random box and hole with visual obs and proprioception +
    target state."""

    def __init__(self):
        super().__init__(
            reset_free=False, **RANDOM_POS_HOLE_KWARGS, **VISUAL_OBS_KWARGS
        )


@register_gym_env("SawyerPegVisual-v2")
class SawyerPegV2VisualInDomain(SawyerPegV2Base):
    """episodic, fixed box and hole with visual obs and proprioception + target
    state."""

    def __init__(self):
        super().__init__(reset_free=False, **FIXED_POS_HOLE_KWARGS, **VISUAL_OBS_KWARGS)


# no visual
@register_gym_env("SawyerPeg-v3")
class SawyerPegV2SmallTableEval(SawyerPegV2Base):
    """episodic, random box and hole.

    small table
    """

    def __init__(self):
        super().__init__(
            reset_free=False,
            small_table=True,
            **RANDOM_POS_HOLE_KWARGS,
            **LOW_DIM_KWARGS,
        )


# visual
@register_gym_env("SawyerPegVisual-v3")
class SawyerPegV2EvalSmallTableVisual(SawyerPegV2Base):
    """episodic, random box and hole with visual obs and proprioception +
    target state.

    small table
    """

    def __init__(self):
        super().__init__(
            reset_free=False,
            small_table=True,
            **RANDOM_POS_HOLE_KWARGS,
            **VISUAL_OBS_KWARGS,
        )


# Reset Free envs
# no visual
@register_gym_env("RFSawyerPegRandomStdMeasure-v1")
class RFSawyerPegV2RandomStdMeasure(SawyerPegV2Base):
    """RF random targets almost reversible, min reward-shaping."""

    def __init__(self):
        super().__init__(
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **LOW_DIM_KWARGS,
        )


# visual
@register_gym_env("RFSawyerPegRandomStdMeasureVisual-v1")
class RFSawyerPegV2RandomStdMeasureVisual(SawyerPegV2Base):
    """RF random targets almost reversible, min reward-shaping, with visual obs
    and proprioception + target state."""

    def __init__(self):
        super().__init__(
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **VISUAL_OBS_KWARGS,
        )


# RF with other state measurements
# no visual
@register_gym_env("RFSawyerPegRandomEuclideanMeasure-v1")
class RFSawyerPegV2RandomEuclideanMeasure(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping."""

    def __init__(self):
        super().__init__(
            state_measure_config="euclidean",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **LOW_DIM_KWARGS,
        )


# visual
@register_gym_env("RFSawyerPegRandomEuclideanMeasureVisual-v1")
class RFSawyerPegV2RandomEuclideanMeasureVisual(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping, with visual obs and proprioception + target state."""

    def __init__(self):
        super().__init__(
            state_measure_config="euclidean",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **VISUAL_OBS_KWARGS,
        )


# no visual
@register_gym_env("RFSawyerPegRandomDTWMeasure-v1")
class RFSawyerPegV2RandomDTWMeasure(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping."""

    def __init__(self):
        super().__init__(
            state_measure_config="dtw",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **LOW_DIM_KWARGS,
        )


# visual
@register_gym_env("RFSawyerPegRandomDTWMeasureVisual-v1")
class RFSawyerPegV2RandomDTWMeasureVisual(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping, with visual obs and proprioception + target state."""

    def __init__(self):
        super().__init__(
            state_measure_config="dtw",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **VISUAL_OBS_KWARGS,
        )


# no visual
@register_gym_env("RFSawyerPegRandomEntropyMeasure-v1")
class RFSawyerPegV2RandomEntropyMeasure(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping."""

    def __init__(self):
        super().__init__(
            state_measure_config="entropy",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **LOW_DIM_KWARGS,
        )


# visual
@register_gym_env("RFSawyerPegRandomEntropyMeasureVisual-v1")
class RFSawyerPegV2RandomEntropyMeasureVisual(SawyerPegV2Base):
    """RF random targets almost reversible by euclidean measure, min reward-
    shaping, with visual obs and proprioception + target state."""

    def __init__(self):
        super().__init__(
            state_measure_config="entropy",
            **RANDOM_TARGETS_KWARGS,
            **STATE_MEASURE_KWARGS,
            **FIXED_POS_HOLE_KWARGS,
            **VISUAL_OBS_KWARGS,
        )
