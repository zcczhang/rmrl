from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TrainingSixteenCombos,
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
    # with cfg.unlocked():
    #     cfg.sampler_kwargs["num_record_processes"] = None  # all

    TARGET_TYPES = TrainingSixteenCombos
    EVAL_TARGET_TYPES = TrainingSixteenCombos

    @classmethod
    def tag(cls) -> str:
        return f"sixteen_combos-random-std"
