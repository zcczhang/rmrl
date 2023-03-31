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

    TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]
    EVAL_TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]

    @classmethod
    def tag(cls) -> str:
        return f"Red_Apple-Stripe_Plate-random-entropy"
