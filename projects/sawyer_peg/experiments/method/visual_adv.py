from allenact_plugins.models.actor_critic_visual import DiscreteActorCriticVisual
from allenact_plugins.sawyer_peg_plugin import (
    SawyerPegMultiDiscretizeEnv,
    MultiDiscretizeSawyerPegTaskSampler,
    TwoPhasesSawyerPegMultiDiscreteTaskSampler,
)
from allenact_plugins.sawyer_peg_plugin.registered_envs import parse_sawyer_peg_env
from allenact_plugins.wrappers import ExperimentConfigBase, DictObsSensor
from projects.sawyer_peg.experiments.visual_base import get_config


CFG = get_config(8, debug=False)
with CFG.unlocked():
    CFG.train_gpus = [0, 1]


class SawyerPegRFVisualTwoPhases(ExperimentConfigBase):

    cfg = CFG

    ENV = parse_sawyer_peg_env(
        [
            "visual_obs_kwargs",
            "two_phases_kwargs",
            "fixed_resets_kwargs",
            "fixed_pos_hole_kwargs",
        ]
    )
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
    TASK_SAMPLER = TwoPhasesSawyerPegMultiDiscreteTaskSampler  # two phase task sampler
    EVAL_TASK_SAMPLER = MultiDiscretizeSawyerPegTaskSampler

    @classmethod
    def tag(cls) -> str:
        return "two_phases-1e4-visual"
