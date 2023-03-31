from projects.strech_manipulation.experiments.tests.budegt_and_measurement.measure_random_in_domain import (
    get_measure_test_cfg,
    N_COMBO,
    StretchMeasureTestConfig,
    MEASURE,
)

from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    STRETCH_MANIPULATHOR_FURNISHED_COMMIT_ID,
)


class SingleComboConfig(StretchMeasureTestConfig):

    cfg = get_measure_test_cfg(
        n_combo=N_COMBO,
        irr_measure_method=MEASURE,
        novel_objects=True,
        extra_tag="novel_scene_test",
        debug=False,
        countertop_scale=None,
        randomize_materials_lighting=True,
        texture_randomization_keys=None,
        cfg_name="rgb.yaml",
    )

    EVAL_TARGET_TYPES = cfg.EVAL_TARGET_TYPES

    THOR_TEST_COMMIT_ID = STRETCH_MANIPULATHOR_FURNISHED_COMMIT_ID
