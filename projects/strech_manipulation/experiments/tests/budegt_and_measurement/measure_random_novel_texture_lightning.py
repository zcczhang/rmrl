from projects.strech_manipulation.experiments.tests.budegt_and_measurement.measure_random_in_domain import (
    StretchMeasureTestConfig,
    get_measure_test_cfg,
    N_COMBO,
    MEASURE,
)


class SingleComboConfig(StretchMeasureTestConfig):

    cfg = get_measure_test_cfg(
        n_combo=N_COMBO,
        irr_measure_method=MEASURE,
        novel_objects=False,
        extra_tag="novel_texture_lightning_test",
        debug=False,
        countertop_scale=None,
        randomize_materials_lighting=True,
        texture_randomization_keys=None,
        cfg_name="rgb.yaml",
    )
    EVAL_TARGET_TYPES = cfg.EVAL_TARGET_TYPES
