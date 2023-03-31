from collections import OrderedDict

import ai2thor.fifo_server

INTEL_CAMERA_WIDTH, INTEL_CAMERA_HEIGHT = 224, 224
STRETCH_MANIPULATHOR_COMMIT_ID = "569fca3d8b46ccb7a33467ba1704c77713adc745"
STRETCH_MANIPULATHOR_FURNISHED_COMMIT_ID = "3c8163e1798366233976ba11e96484ac182863d1"
STRETCH_FOV = 69
STRETCH_VISIBILITY_DISTANCE = 1.5

STRETCH_ENV_ARGS = dict(
    gridSize=0.25,
    width=INTEL_CAMERA_WIDTH,
    height=INTEL_CAMERA_HEIGHT,
    visibilityDistance=STRETCH_VISIBILITY_DISTANCE,  # 1.0
    fieldOfView=STRETCH_FOV,
    # agentControllerType="mid-level",
    server_class=ai2thor.fifo_server.FifoServer,
    useMassThreshold=True,
    massThreshold=10,
    autoSimulation=False,
    autoSyncTransforms=True,
    renderInstanceSegmentation=False,
    agentMode="stretch",
    renderDepthImage=False,
    snapToGrid=False,
    commit_id=STRETCH_MANIPULATHOR_COMMIT_ID,
    fixed_camera=True,
)
# if depth to rgb:
KINECT_REAL_W, KINECT_REAL_H = 720, 1280
KINECT_RESIZED_W, KINECT_RESIZED_H = 180, 320
KINECT_FOV_W, KINECT_FOV_H = 59, 90

# if rgb to depth:
# KINECT_REAL_W, KINECT_REAL_H = 576, 640
# KINECT_RESIZED_W, KINECT_RESIZED_H = 288, 320
# KINECT_FOV_W, KINECT_FOV_H = 65, 75

INTEL_REAL_W, INTEL_REAL_H = 1920, 1080
INTEL_RESIZED_W, INTEL_RESIZED_H = 320, 180
INTEL_FOV_W, INTEL_FOV_H = 69, 42

# MIN_INTEL_DEPTH = 0.28
MIN_INTEL_DEPTH = 0
MAX_INTEL_DEPTH = 3
# MIN_KINECT_DEPTH = 0.5
MIN_KINECT_DEPTH = 0.0
MAX_KINECT_DEPTH = 3.86

MOVE_AHEAD = "MoveAhead"
MOVE_BACK = "MoveBack"
MOVE_ARM_HEIGHT_P = "MoveArmHeightP"
MOVE_ARM_HEIGHT_M = "MoveArmHeightM"
MOVE_ARM_Z_P = "MoveArmZP"
MOVE_ARM_Z_M = "MoveArmZM"
PICKUP = "PickUp"
RELEASE = "ReleaseObject"
MOVE_WRIST_P = "MoveWristP"
MOVE_WRIST_M = "MoveWristM"

MOVE_WRIST_P_SMALL = "MoveWristPSmall"
MOVE_WRIST_M_SMALL = "MoveWristMSmall"

ROTATE_LEFT = "RotateLeft"
ROTATE_RIGHT = "RotateRight"
ROTATE_RIGHT_SMALL = "RotateRightSmall"
ROTATE_LEFT_SMALL = "RotateLeftSmall"
DONE = "Done"  # not used

STRETCH_ACTIONS = (
    MOVE_AHEAD,
    MOVE_BACK,
    MOVE_ARM_HEIGHT_P,
    MOVE_ARM_HEIGHT_M,
    MOVE_ARM_Z_P,
    MOVE_ARM_Z_M,
    PICKUP,
    RELEASE,
    MOVE_WRIST_P_SMALL,
    MOVE_WRIST_M_SMALL,
)

STRETCH_ACTIONS_EXTENSION = (
    *STRETCH_ACTIONS,
    MOVE_WRIST_P,
    MOVE_WRIST_M,
)


STRETCH_ACTIONS_FULL = (
    *STRETCH_ACTIONS_EXTENSION,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    ROTATE_RIGHT_SMALL,
    ROTATE_LEFT_SMALL,
)


ADDITIONAL_ARM_ARGS = {
    "disableRendering": True,
    "returnToStart": True,  # False,
    "speed": 1,
}

ARM_LENGTH = 1.0
ARM_MOVE_CONSTANT = 0.05
WRIST_ROTATION = 10  # 1/5 for `*_Small`
AGENT_MOVEMENT_CONSTANT = 0.05
AGENT_ROTATION_DEG = 30

REWARD_CONFIG = {
    "step_penalty": -0.01,
    "failed_action_penalty": -0.03,
    "first_picked_up_reward": 1.0,
    "arm_dist_multiplier": 1.0,
    "obj_dist_multiplier": 1.0,
    "goal_success_reward": 10.0,
}


# relative to agent, use `convert_agent_to_world_coordinate(limit, agent)` to get world coord
# not consider the wrist (so theoretically a little smaller than the reality)
RELATIVE_HAND_BOUNDS = OrderedDict(
    xlow=0.34840691089630127,
    xhigh=1.154906988143921,
    ylow=-0.8690925240516663,
    yhigh=0.2319074273109436,
    zlow=-0.02385216951370247,
    zhigh=-0.02385216951370247,
)

RELATIVE_HAND_BOUNDS_DICT = dict(
    low=dict(
        x=RELATIVE_HAND_BOUNDS["xlow"],
        y=RELATIVE_HAND_BOUNDS["ylow"],
        z=RELATIVE_HAND_BOUNDS["zlow"],
    ),
    high=dict(
        x=RELATIVE_HAND_BOUNDS["xhigh"],
        y=RELATIVE_HAND_BOUNDS["yhigh"],
        z=RELATIVE_HAND_BOUNDS["zhigh"],
    ),
)

"""Partitions"""


def _get_partitions(pick_objs: list, receptacles: list):
    return [(x, y) for x in pick_objs for y in receptacles]


TrainingOneCombo = [("Red_Apple", "Stripe_Plate")]
TrainingFourCombos = _get_partitions(
    ["Red_Apple", "Bread"], ["Stripe_Plate", "Bake_Pan"]
)
TrainingSixteenCombos = _get_partitions(
    ["Red_Apple", "Bread", "Sponge", "Blue_Cube"],
    ["Stripe_Plate", "Bake_Pan", "Metal_Bowl", "Saucer"],
)

TestingCombos = _get_partitions(
    ["Green_Apple", "Tomato", "Potato", "Mug"],
    ["Wooden_Bowl", "Bamboo_Box", "Rusty_Pan", "Pot"],
)

ExpRoomPickObjs = [
    "Red_Apple",
    "Bread",
    "Sponge",
    "Red_Cube",
    "Green_Apple",
    "Potato",
    "Tomato",
    "Blue_Cube",
    "Yellow_Cube",
    "Mug",
]
ExpRoomContainers = [
    "Stripe_Plate",
    "Bake_Pan",
    "Medal_Bowl",
    "Pot",
    "Wooden_Bowl",
    "Bamboo_Box",
    "Rusty_Pan",
    "Saucer",
]
ExpRoomCombo = _get_partitions(ExpRoomPickObjs, ExpRoomContainers)

# PEG_OBJS = [
#     "Yellow_Square_Peg",
#     "Star_Peg",
#     "Wooden_Square_Peg",
#     "Red_Square_Peg",
# ]
#
# PEG_CONTAINERS = [
#     "Red_Square_Peg_Box",
#     "Square_Peg_Box__3",
#     "Wooden_Star_Peg_Box",
#     "Square_Peg_Box__2",
#     "Wooden_Peg_Box",
# ]


# For state irreversible measure
STATE_MEASURE_CONFIGS = dict(
    std=dict(
        history=dict(hand=[], obj=[]),
        thresholds=dict(hand=0.1, obj=0.02, num_phases_irr_tolerance=2),
        metrics=dict(name="std", hand=0.0, obj=0.0),
    ),
    entropy=dict(
        history=dict(hand=[], obj=[]),
        thresholds=dict(hand=4, obj=0.5, num_phases_irr_tolerance=2),
        metrics=dict(name="entropy", hand=None, obj=None),
        entropy_fn="scipy",  # scipy, npeet, npeetd
        grid_size=[40, 20, 20],
    ),
    euclidean=dict(
        history=dict(hand=[], obj=[]),
        thresholds=dict(hand=0.2, obj=0.005),
        metrics=dict(name="euclidean", hand=None, obj=None),
        time_horizons=dict(memory=500, measure_steps=100),
    ),
    dtw=dict(
        history=dict(hand=[], obj=[]),
        thresholds=dict(hand=0.2, obj=0.015),
        metrics=dict(name="dtw", hand=None, obj=None),
        time_horizons=dict(memory=500, measure_steps=100),
    ),
)

__all__ = [C for C in dir() if C[0].isupper()]
