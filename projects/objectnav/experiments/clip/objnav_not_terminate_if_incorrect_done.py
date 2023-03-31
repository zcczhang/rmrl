from projects.objectnav.experiments.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
    get_config,
)


CFG = get_config(debug=False, reset_free=False, done_always_terminate=False)


class ObjNavEpisodic(ObjectNavRoboThorClipRGBPPOExperimentConfig):
    cfg = CFG

    def __init__(self):
        # TODO mv to constants
        reward_config = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": -0.05,  # TODO
            "reached_max_steps_reward": 0.0,
            "shaping_weight": 1.0,
        }
        super().__init__(reward_config=reward_config)

    MAX_STEPS = 300  # 500

    @classmethod
    def tag(cls) -> str:
        return (
            "rf10-objnav-fixed_phase_length_unless_success-fail_step-0.05-maxstep=300"
        )
