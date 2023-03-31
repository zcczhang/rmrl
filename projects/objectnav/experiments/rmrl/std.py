from projects.objectnav.experiments.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
    get_config,
)


CFG = get_config(
    debug=False,
    done_always_terminate=False,
    reset_free=True,
    measurement_lead_reset=True,
    measure_method="std",
)


class ObjNavStdMeasureRMRL(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    cfg = CFG

    def __init__(self):
        reward_config = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": -0.05,
            "reached_max_steps_reward": 0.0,
            "shaping_weight": 1.0,
        }
        super().__init__(reward_config=reward_config)

    MAX_STEPS = 300

    @classmethod
    def tag(cls) -> str:
        return "objnav-std-0.1"
