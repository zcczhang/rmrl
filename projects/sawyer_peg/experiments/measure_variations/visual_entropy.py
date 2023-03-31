from allenact_plugins.sawyer_peg_plugin.registered_envs import parse_sawyer_peg_env
from allenact_plugins.wrappers import DictObsSensor
from ..visual_base import SawyerPegRFRandomAlmostReversibleVisual, get_config


class SawyerPegRFRandomAlmostReversibleVisualentropy(
    SawyerPegRFRandomAlmostReversibleVisual
):
    processes = 8
    cfg = get_config(processes, debug=False)
    ENV = parse_sawyer_peg_env("visual_entropy_measure")
    input_uuid = "sawyer_peg_obs"
    SENSORS = [
        DictObsSensor(
            uuid=input_uuid,
            num_stacked_frames=cfg.model_kwargs["num_stacked_frames"],
            env_name=ENV,
        ),
    ]
    MEASUREMENT = "entropy"
