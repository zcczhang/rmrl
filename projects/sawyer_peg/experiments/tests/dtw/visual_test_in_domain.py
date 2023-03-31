from allenact_plugins.sawyer_peg_plugin.registered_envs import parse_sawyer_peg_env
from projects.sawyer_peg.experiments.visual_base import (
    SawyerPegRFRandomAlmostReversibleVisual,
)


class SawyerPegRFRandomAlmostReversibleVisualBaseline(
    SawyerPegRFRandomAlmostReversibleVisual
):
    TEST_ENV = parse_sawyer_peg_env("visual_in_domain_eval")
    MEASUREMENT = "dtw"

    @classmethod
    def tag(cls) -> str:
        return super().tag() + "_test_in_domain"
