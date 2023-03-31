from typing import Sequence, Union, Optional

import gym
import yaml

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from allenact_plugins.navigation_plugin.objectnav.models import (
    ResnetTensorNavActorCritic,
)
from allenact_plugins.robothor_plugin.robothor_constants import (
    DEFAULT_REWARD_CONFIG,
    OBJ_NAV_TARGET_TYPES,
    OBJNAV_DEFAULT_ENV_ARGS,
)
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_task_samplers import (
    ObjectNavDatasetTaskSampler,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from allenact_plugins.utils import get_file, Config
from allenact_plugins.wrappers import ThorExperimentConfigBase
from .mixins import ClipResNetPreprocessorMixin

GOAL_OBJ_TYPE_UUID = "obj_type_goal_uuid"


def get_config(
    debug: bool = False,
    *,
    done_always_terminate: bool = False,
    reset_free: bool = False,
    measurement_lead_reset: Optional[bool] = None,
    measure_method: Optional[str] = None,
    num_steps_for_reset: Optional[int] = None,
):
    cfg_path = yaml.safe_load(get_file(__file__, "cfg", "clip_resnet50_gru.yaml"))
    cfg = Config(cfg_path)
    if debug:
        cfg.debug()
        cfg.training_setting_kwargs["metric_accumulate_interval"] = 1
    with cfg.unlocked():
        cfg.model_kwargs["goal_sensor_uuid"] = GOAL_OBJ_TYPE_UUID

        cfg.sampler_kwargs["task_args"]["done_always_terminate"] = done_always_terminate
        cfg.valid_sampler_kwargs["task_args"][
            "done_always_terminate"
        ] = done_always_terminate

        cfg.sampler_kwargs["reset_free"] = reset_free
        if reset_free:
            cfg.sampler_kwargs["measurement_lead_reset"] = measurement_lead_reset
            cfg.sampler_kwargs["measure_method"] = measure_method
            cfg.sampler_kwargs["num_steps_for_reset"] = num_steps_for_reset
    return cfg


class ObjectNavRoboThorClipRGBPPOExperimentConfig(ThorExperimentConfigBase):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    cfg_path = yaml.safe_load(get_file(__file__, "cfg", "clip_resnet50_gru.yaml"))
    cfg = get_config(debug=False, reset_free=False, done_always_terminate=True)

    SCREEN_SIZE = 224
    MAX_STEPS = 300  # 500

    TARGET_TYPES = OBJ_NAV_TARGET_TYPES
    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(object_types=TARGET_TYPES, uuid=GOAL_OBJ_TYPE_UUID),
    ]

    ENV_CLASS = RoboThorEnvironment
    TASK_CLASS = ObjectNavTask
    ACTION_SPACE = gym.spaces.Discrete(len(ObjectNavTask.class_action_names()))
    TASK_SAMPLER = ObjectNavDatasetTaskSampler
    MODEL = ResnetTensorNavActorCritic

    CLIP_MODEL_TYPE = "RN50"

    def __init__(
        self, reward_config=DEFAULT_REWARD_CONFIG, env_args=OBJNAV_DEFAULT_ENV_ARGS
    ):
        super().__init__(reward_config=reward_config, env_args=env_args)
        self.preprocessing = ClipResNetPreprocessorMixin(
            sensors=self.SENSORS,
            rgb_output_uuid=self.cfg.model_kwargs["rgb_resnet_preprocessor_uuid"],
            depth_output_uuid=self.cfg.model_kwargs["depth_resnet_preprocessor_uuid"],
            clip_model_type=self.CLIP_MODEL_TYPE,
        )

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return self.preprocessing.preprocessors()

    @property
    def scenes_sub_dir(self):
        return "episodes"

    @property
    def data_suffix(self):
        return ".json.gz"

    @classmethod
    def tag(cls) -> str:
        return "rf11-episodic-objnav-latest_commit"
