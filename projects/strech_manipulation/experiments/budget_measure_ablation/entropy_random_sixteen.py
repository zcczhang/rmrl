from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TrainingSixteenCombos,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(
    debug=False,
    countertop_scale=None,
    keep_extra_furniture=False,
    randomize_materials_lighting=False,
    texture_randomization_keys=None,
    irr_measure_method="entropy",
    cfg_name="rgb.yaml",
)


class SingleComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = TrainingSixteenCombos
    EVAL_TARGET_TYPES = TrainingSixteenCombos

    @classmethod
    def tag(cls) -> str:
        return f"Red_Apple-Stripe_Plate-random-entropy"
