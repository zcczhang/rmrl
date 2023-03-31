import numpy as np


STATE_MEASURE_CONFIGS = dict(
    std=dict(
        history=dict(tcp_center=[], obj_pos=[]),
        thresholds=dict(tcp_center=0.01, obj_pos=0.001, num_phases_irr_tolerance=2),
        metrics=dict(name="std", tcp_center=0.0, obj_pos=0.0),
    ),
    euclidean=dict(
        history=dict(tcp_and_obj=[]),
        thresholds=dict(tcp_and_obj=0.01),  # 0.01 for max(mean()) is good
        metrics=dict(name="euclidean", tcp_and_obj=0.0),
        time_horizons=dict(memory=500, measure_steps=100),
        schedule=dict(min=500, max=1000, schedule_steps=1e4),  # not necessary tho
    ),
    dtw=dict(
        history=dict(tcp_and_obj=[]),
        thresholds=dict(tcp_and_obj=0.05),
        metrics=dict(name="dtw", tcp_and_obj=0.0),
        time_horizons=dict(memory=500, measure_steps=100),
    ),
    entropy=dict(
        history=dict(tcp_and_obj=[]),
        thresholds=dict(tcp_and_obj=1.4, num_phases_irr_tolerance=2),
        metrics=dict(name="entropy", tcp_and_obj=0.0),
        grid_size=[20, 20, 10],
    ),
)

BASIC_SPACE = dict(
    obj_low=(0.0, 0.5, 0.02),
    obj_high=(0.2, 0.7, 0.02),
    goal_low=(-0.35, 0.5, -0.001),
    goal_high=(-0.25, 0.7, 0.001),
)
WIDER_SPACE = dict(
    obj_low=(-0.3, 0.35, 0.02),
    obj_high=(0.1, 0.85, 0.02),
    goal_low=(-0.525, 0.3, -0.001),  # 0.4
    goal_high=(-0.525, 0.75, 0.001),  # 0.7
)

COMPUTER_REWARD_KWARGS_FULL = dict(
    obj_to_target_reward_coef=100.0,
    tcp_to_obj_reward_coef=100.0,
    gripper_width_reward_coef=2.0,
    z_reward_coef=2.0,
    use_z_reward=True,
    obj_to_target_scale=np.array([1.0, 2.0, 2.0]),
    use_first_grasped_reward=True,
    use_gripper_reward=True,
)
COMPUTER_REWARD_KWARGS_MIN = dict(
    obj_to_target_reward_coef=100.0,
    tcp_to_obj_reward_coef=100.0,
    use_z_reward=False,
    obj_to_target_scale=1,
    use_first_grasped_reward=False,
    use_gripper_reward=False,
)
POSSIBLE_REWARD_KWARGS = dict(
    full=COMPUTER_REWARD_KWARGS_FULL, min=COMPUTER_REWARD_KWARGS_MIN
)


# Copy from EARL
INITIAL_STATES = np.array(
    [
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.00313463, 0.68326396, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, -0.04035005, 0.67949003, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.02531051, 0.6074387, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.05957219, 0.6271171, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, -0.07566337, 0.62575287, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, -0.01177235, 0.55206996, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.02779735, 0.54707706, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.01835314, 0.5329686, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.02690855, 0.6263067, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.01766127, 0.59630984, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.0560186, 0.6634998, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, -0.03950658, 0.6323736, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, -0.03216827, 0.5247563, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.01265727, 0.69466716, 0.02],
        [0.00615235, 0.6001898, 0.19430117, 1.0, 0.05076993, 0.6025737, 0.02],
    ]
)

GOAL_STATES = np.array([[0.0, 0.6, 0.2, 1.0, -0.3 + 0.03, 0.6, 0.0 + 0.13]])
