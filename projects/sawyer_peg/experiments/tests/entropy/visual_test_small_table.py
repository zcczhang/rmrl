from allenact_plugins.sawyer_peg_plugin.registered_envs import parse_sawyer_peg_env
from projects.sawyer_peg.experiments.visual_base import (
    SawyerPegRFRandomAlmostReversibleVisual,
)


class SawyerPegRFRandomAlmostReversibleVisualBaseline(
    SawyerPegRFRandomAlmostReversibleVisual
):
    TEST_ENV = parse_sawyer_peg_env("visual_small_table")
    MEASUREMENT = "entropy"

    @classmethod
    def tag(cls) -> str:
        return super().tag() + "_test_small_table"
