from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TrainingFourCombos,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(debug=False, countertop_scale=None, keep_extra_furniture=False)
with CFG.unlocked():
    CFG.sampler_kwargs["irr_measure"] = False
    CFG.sampler_kwargs["num_steps_for_resets"] = 1e3


class SingleComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = TrainingFourCombos
    EVAL_TARGET_TYPES = TrainingFourCombos

    @classmethod
    def tag(cls) -> str:
        return f"FourCombos-1e3-random"
