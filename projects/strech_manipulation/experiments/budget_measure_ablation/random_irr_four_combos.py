from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TrainingFourCombos,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)


class FourComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = get_cfg(
        debug=False,
        countertop_scale=None,
        keep_extra_furniture=False,
        randomize_materials_lighting=False,
        texture_randomization_keys=None,
    )

    TARGET_TYPES = TrainingFourCombos
    EVAL_TARGET_TYPES = TrainingFourCombos

    @classmethod
    def tag(cls) -> str:
        return f"four_combos-random-std"
