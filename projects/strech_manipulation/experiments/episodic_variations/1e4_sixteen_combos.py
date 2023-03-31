from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TrainingSixteenCombos,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(debug=False, countertop_scale=None, keep_extra_furniture=False)
with CFG.unlocked():
    CFG.sampler_kwargs["irr_measure"] = False
    CFG.sampler_kwargs["num_steps_for_resets"] = 1e4


class SixteenComboConfigTwoPhasesRandom(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = TrainingSixteenCombos
    EVAL_TARGET_TYPES = TrainingSixteenCombos

    @classmethod
    def tag(cls) -> str:
        return f"SixteenCombos-1e4-random"
