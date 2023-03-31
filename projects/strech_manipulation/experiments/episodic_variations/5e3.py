from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(debug=False, countertop_scale=None, keep_extra_furniture=False)
with CFG.unlocked():
    CFG.sampler_kwargs["irr_measure"] = False
    CFG.sampler_kwargs["num_steps_for_resets"] = 5e3


class SingleComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]
    EVAL_TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]

    @classmethod
    def tag(cls) -> str:
        return f"Red_Apple-Stripe_Plate-5e3-random"
