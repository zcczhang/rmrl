from allenact_plugins.ithor_plugin.ithor_util import horizontal_to_vertical_fov

MOVE_AHEAD = "MoveAhead"
ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
LOOK_DOWN = "LookDown"
LOOK_UP = "LookUp"
END = "End"
PASS = "Pass"

CAMERA_WIDTH = 400
CAMERA_HEIGHT = 300
HORIZONTAL_FIELD_OF_VIEW = 79


DEFAULT_REWARD_CONFIG = {
    "step_penalty": -0.01,
    "goal_success_reward": 10.0,
    "failed_stop_reward": 0.0,  # TODO for RF/RM
    "reached_max_steps_reward": 0.0,
    "shaping_weight": 1.0,
}

THOR_COMMIT_ID = "5ca1d87decf19ff0d473d4430029a2f3bcd2eaab"
OBJNAV_DEFAULT_ENV_ARGS = dict(
    width=CAMERA_WIDTH,
    height=CAMERA_HEIGHT,
    continuousMode=True,
    applyActionNoise=True,
    rotateStepDegrees=30.0,
    visibilityDistance=1.0,  # 1.5,
    gridSize=0.25,
    snapToGrid=False,
    agentMode="locobot",
    fieldOfView=horizontal_to_vertical_fov(
        horizontal_fov_in_degrees=HORIZONTAL_FIELD_OF_VIEW,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    ),
    include_private_scenes=False,
    commit_id=THOR_COMMIT_ID,
)

OBJ_NAV_TARGET_TYPES = tuple(
    sorted(
        [
            "AlarmClock",
            "Apple",
            "BaseballBat",
            "BasketBall",
            "Bowl",
            "GarbageCan",
            "HousePlant",
            "Laptop",
            "Mug",
            "SprayBottle",
            "Television",
            "Vase",
        ]
    )
)


# For state irreversible measure
STATE_MEASURE_CONFIGS = dict(
    std=dict(
        history=dict(agent=[]),
        thresholds=dict(agent=0.1, num_phases_irr_tolerance=2),
        metrics=dict(name="std", agent=None),
    ),
    euclidean=dict(
        history=dict(agent=[]),
        thresholds=dict(agent=0.1),
        metrics=dict(name="euclidean", agent=None),
        time_horizons=dict(memory=500, measure_steps=100),
    ),
)
