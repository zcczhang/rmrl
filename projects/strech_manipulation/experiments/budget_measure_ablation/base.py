from typing import Optional, Union, Tuple

import gym
import yaml

from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.models.visuomotor_with_prompt import (
    DiscreteActorCriticVisualWithPrompt,
)
from allenact_plugins.stretch_manipulathor_plugin.strech_sensors import (
    RGBSensorStretch,
    StretchPromptSensor,
    StretchPolarSensor,
)
from allenact_plugins.stretch_manipulathor_plugin.strech_task_sampler_exproom import (
    StretchExpRoomPickPlaceTaskSampler,
    StretchExpRoomPickPlaceResetFreeTaskSampler,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_arm_environment import (
    StretchManipulaTHOREnvironment,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    STRETCH_MANIPULATHOR_COMMIT_ID,
    REWARD_CONFIG,
    STRETCH_ENV_ARGS,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_tasks import StretchPickPlace
from allenact_plugins.utils import get_file, Config
from allenact_plugins.wrappers import ThorExperimentConfigBase

RGB_OBS_KEYS = ("wrist",)
STATE_OBS_KEYS = ("hand", "target")

RGB_INPUT_ID = "rgb"
PROMPT_INPUT_ID = "prompt"
LOW_DIM_INPUT_ID = "state"
PROMPT_IMG_UUID = "prompt_img"
ACTION_AFFORDANCE_UUID = "action_affordance"

PROMPT_GOAL = True

# No need
USE_ACTION_AFFORDANCE = False
PREV_ACTION_EMBED_DIM: Optional[int] = None


def get_cfg(
    debug: bool = False,
    *,
    # affect eval kwargs, default for training & in-domain positional testing
    randomize_materials_lighting: bool = False,
    texture_randomization_keys: Optional[Union[str, Tuple[str, ...]]] = None,
    countertop_scale: Optional[Union[int, float, list, tuple]] = None,
    keep_extra_furniture: bool = False,  # deprecate
    irr_measure_method: Optional[str] = "std",
    num_steps_for_resets: Optional[int] = None,
    random_targets: bool = True,
    reset_if_obj_dropped: bool = False,
    cfg_name: str = "rgb.yaml",
):
    cfg_path = yaml.safe_load(get_file(__file__, "cfg", cfg_name))
    cfg = Config(cfg_path)
    if debug:
        cfg.debug()
    with cfg.unlocked():
        cfg.model_kwargs["prompt_img_uuid"] = PROMPT_IMG_UUID
        # just to make preprocessors as a direct acyclic graph
        cfg.model_kwargs["rgb_uuid"] = RGB_INPUT_ID
        cfg.model_kwargs["prompt_uuid"] = PROMPT_INPUT_ID
        cfg.model_kwargs["goal_uuid"] = None
        cfg.model_kwargs["state_uuid"] = LOW_DIM_INPUT_ID
        # no need at all
        # CFG.model_kwargs["obj_in_hand_uuid"] = "pickup_sensor"
        if USE_ACTION_AFFORDANCE:
            cfg.model_kwargs["extra_discrete_sensor_uuid"] = ACTION_AFFORDANCE_UUID
            cfg.model_kwargs["extra_discrete_sensor_dim"] = 2
            cfg.model_kwargs["extra_discrete_sensor_embedding_dim"] = 128
        if PREV_ACTION_EMBED_DIM is not None:
            cfg.model_kwargs["condition_on_prev_action"] = True
            cfg.model_kwargs["action_embed_dim"] = PREV_ACTION_EMBED_DIM

        if isinstance(irr_measure_method, str):
            cfg.sampler_kwargs["irr_measure"] = True
            cfg.sampler_kwargs["irr_measure_method"] = irr_measure_method
        else:
            assert num_steps_for_resets is not None
            cfg.sampler_kwargs["irr_measure"] = False
            cfg.sampler_kwargs["num_steps_for_resets"] = num_steps_for_resets
        if random_targets:
            cfg.sampler_kwargs["random_targets"] = True
            cfg.sampler_kwargs["two_phase"] = False
        else:
            cfg.sampler_kwargs["two_phase"] = True
            cfg.sampler_kwargs["random_targets"] = False

        cfg.sampler_kwargs["reset_if_obj_dropped"] = reset_if_obj_dropped

        if randomize_materials_lighting:
            cfg.valid_sampler_kwargs["spec_env_kwargs"]["randomize_materials"] = True
            cfg.valid_sampler_kwargs["spec_env_kwargs"]["randomize_lighting"] = True
            cfg.valid_sampler_kwargs["spec_env_kwargs"][
                "controllable_randomization"
            ] = True
            if texture_randomization_keys is None:
                texture_randomization_keys = ("table", "wall", "floor")
            cfg.valid_sampler_kwargs[
                "texture_randomization_keys"
            ] = texture_randomization_keys
        cfg.valid_sampler_kwargs["countertop_scale"] = countertop_scale
        cfg.valid_sampler_kwargs["keep_extra_furniture"] = keep_extra_furniture

    return cfg


class RFStretchPickPlaceExperimentConfigBase(ThorExperimentConfigBase):

    cfg = get_cfg()

    SCREEN_SIZE = 224
    THOR_COMMIT_ID = STRETCH_MANIPULATHOR_COMMIT_ID

    MAX_STEPS = 300

    TARGET_TYPES = None
    EVAL_TARGET_TYPES = None

    input_uuid = None
    SENSORS = [
        RGBSensorStretch(
            rgb_keys=RGB_OBS_KEYS,
            concatenate_if_multiple=True,
            uuid=RGB_INPUT_ID,
            width=SCREEN_SIZE * len(RGB_OBS_KEYS),
            height=SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        StretchPolarSensor(uuid=LOW_DIM_INPUT_ID, state_obs_key=STATE_OBS_KEYS,),
        StretchPromptSensor(uuid=PROMPT_INPUT_ID),
        # StretchObjInHandSensor(uuid="pickup_sensor"),
        # StretchActionAffordanceSensor(uuid=ACTION_AFFORDANCE_UUID),
    ]
    TASK_CLASS = StretchPickPlace
    ENV_CLASS = StretchManipulaTHOREnvironment
    ACTION_SPACE = gym.spaces.Discrete(len(TASK_CLASS.class_action_names()))
    TASK_SAMPLER = StretchExpRoomPickPlaceResetFreeTaskSampler
    EVAL_TASK_SAMPLER = StretchExpRoomPickPlaceTaskSampler
    MODEL = DiscreteActorCriticVisualWithPrompt

    def __init__(
        self, reward_config: dict = REWARD_CONFIG, env_args: dict = STRETCH_ENV_ARGS
    ):
        super().__init__(reward_config=reward_config, env_args=env_args)

    def preprocessors(self):
        return [
            ClipResNetPreprocessor(
                rgb_input_uuid=RGB_INPUT_ID,
                prompt_input_uuid=PROMPT_INPUT_ID,
                clip_model_type="RN50",
                pool=False,
                output_uuid=PROMPT_IMG_UUID,
                input_img_height_width=(
                    self.SCREEN_SIZE,
                    self.SCREEN_SIZE * len(RGB_OBS_KEYS),
                ),
                normalize_img_features=False,
                normalize_text_features=False,
            )
        ]

    @classmethod
    def tag(cls) -> str:
        raise NotImplementedError
