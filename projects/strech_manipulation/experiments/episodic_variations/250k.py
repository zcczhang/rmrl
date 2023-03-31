from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)

CFG = get_cfg(debug=False, countertop_scale=None, keep_extra_furniture=False)
with CFG.unlocked():
    CFG.sampler_kwargs["irr_measure"] = False
    CFG.sampler_kwargs["num_steps_for_resets"] = 2.5e4
    CFG.sampler_kwargs["reset_if_obj_dropped"] = True

    CFG.validation_tasks = 1
    CFG.loss_steps = 60000000
    CFG.training_setting_kwargs["save_interval"] = 10000000  # 10M


class SingleComboConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = CFG

    TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]
    EVAL_TARGET_TYPES = [("Red_Apple", "Stripe_Plate")]

    @classmethod
    def tag(cls) -> str:
        return (
            f"{cls.TARGET_TYPES[0][0]}-{cls.TARGET_TYPES[0][1]}-250k-reset_if_dropped"
        )
