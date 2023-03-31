import yaml

from allenact_plugins.models.actor_critic_visual import DiscreteActorCriticVisual
from allenact_plugins.sawyer_peg_plugin import (
    SawyerPegMultiDiscretizeEnv,
    MultiDiscretizeSawyerPegTaskSampler,
)
from allenact_plugins.sawyer_peg_plugin.registered_envs import parse_sawyer_peg_env
from allenact_plugins.utils import Config
from allenact_plugins.utils.env_utils import get_file
from allenact_plugins.wrappers import ExperimentConfigBase, DictObsSensor

__all__ = ["get_config", "SawyerPegRFRandomAlmostReversibleVisual"]


def get_config(proc: int = 8, debug: bool = False):
    if proc == 8:
        cfg_name = "visual.yaml"
    elif proc == 4:
        cfg_name = "visual_4_proc.yaml"
    else:
        raise NotImplementedError(proc)
    cfg_path = yaml.safe_load(get_file(__file__, "cfg", cfg_name))
    cfg = Config(cfg_path)
    if debug:
        cfg.debug()
    return cfg


class SawyerPegRFRandomAlmostReversibleVisual(ExperimentConfigBase):

    processes = 8
    cfg = get_config(processes, debug=False)
    ENV = parse_sawyer_peg_env("visual_train")
    EVAL_ENV = parse_sawyer_peg_env("visual_in_domain_eval")
    TEST_ENV = parse_sawyer_peg_env("visual_eval")
    input_uuid = "sawyer_peg_obs"
    SENSORS = [
        DictObsSensor(
            uuid=input_uuid,
            num_stacked_frames=cfg.model_kwargs["num_stacked_frames"],
            env_name=ENV,
        ),
    ]
    MODEL = DiscreteActorCriticVisual
    ENV_WRAPPER = SawyerPegMultiDiscretizeEnv
    TASK_SAMPLER = MultiDiscretizeSawyerPegTaskSampler

    MEASUREMENT = "std"

    @classmethod
    def tag(cls) -> str:
        return (
            f"{cls.MEASUREMENT}_measure-random_target-visual-{cls.processes}_processes"
        )
