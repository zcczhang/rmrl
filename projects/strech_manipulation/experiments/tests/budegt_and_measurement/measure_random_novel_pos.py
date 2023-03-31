from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    TestingCombos,
    TrainingFourCombos,
    TrainingSixteenCombos,
)
from projects.strech_manipulation.experiments.budget_measure_ablation.base import (
    RFStretchPickPlaceExperimentConfigBase,
    get_cfg,
)


# FILL BUDGET AND MEASURE HERE
N_COMBO = 1
MEASURE = "std"


def get_measure_test_cfg(
    *,
    n_combo: int = 1,
    irr_measure_method: str,
    novel_objects: bool,
    extra_tag: str,
    **kwargs,
):
    cfg = get_cfg(irr_measure_method=irr_measure_method, **kwargs)
    combo_map = {
        1: [("Red_Apple", "Stripe_Plate")],
        4: TrainingFourCombos,
        16: TrainingSixteenCombos,
    }
    with cfg.unlocked():
        if not novel_objects:
            cfg.EVAL_TARGET_TYPES = combo_map[n_combo]
        else:
            cfg.EVAL_TARGET_TYPES = TestingCombos

        cfg.tag = f"{n_combo}Combo-random-{irr_measure_method}-{extra_tag}"

    return cfg


class StretchMeasureTestConfig(RFStretchPickPlaceExperimentConfigBase):

    cfg = get_measure_test_cfg(
        n_combo=N_COMBO,
        irr_measure_method=MEASURE,
        novel_objects=False,
        extra_tag="pos_ood_test",
        debug=False,
        countertop_scale=None,
        randomize_materials_lighting=False,
        texture_randomization_keys=None,
        cfg_name="rgb.yaml",
    )

    EVAL_TARGET_TYPES = cfg.EVAL_TARGET_TYPES  # type: ignore

    @classmethod
    def tag(cls) -> str:
        return f"{cls.cfg.tag}"
