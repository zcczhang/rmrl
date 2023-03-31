from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TestingCombos,
    STRETCH_MANIPULATHOR_FURNISHED_COMMIT_ID,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(
    debug=False,
    countertop_scale=None,
    keep_extra_furniture=False,
    randomize_materials_lighting=True,
    texture_randomization_keys=None,
    irr_measure_method="euclidean",
    cfg_name="rgb.yaml",
)


class SingleComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]
    EVAL_TARGET_TYPES = TestingCombos
    THOR_TEST_COMMIT_ID = STRETCH_MANIPULATHOR_FURNISHED_COMMIT_ID
    MAX_STEPS = 300

    @classmethod
    def tag(cls) -> str:
        return f"Red_Apple-Stripe_Plate-random-5e3-novel_scene_test"
