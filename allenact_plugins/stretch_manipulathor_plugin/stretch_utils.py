import random
from typing import Optional, Union, Literal

import numpy as np
import quaternion
from ai2thor.controller import Controller
from scipy.spatial.transform import Rotation as R

from allenact.utils.system import get_logger
from allenact_plugins.utils import bbox_intersect

__all__ = [
    "make_all_objects_unbreakable",
    "remove_nan_inf_for_frames",
    "get_relative_stretch_current_arm_state",
    "position_distance",
    "close_enough",
    "get_agent_teleport_kwargs_from_metadata",
    "object_ids_to_types",
    "compute_pointgoal",
    "get_polar_pos",
    "sample_obj_pos_over_space",
    "make_rotation_matrix",
    "inverse_rot_trans_mat",
    "position_rotation_from_mat",
    "find_closest_inverse",
    "calc_inverse",
    "get_rel_obj1_to_obj2",
    "get_abs_obj1_from_rel_obj1_to_obj2",
    "get_abs_obj2_from_abs_obj1_and_rel_obj1_to_obj2",
    "CONSTANTLY_MOVING_OBJECTS",
]


def make_all_objects_unbreakable(controller: Controller):
    all_breakable_objects = [
        o["objectType"]
        for o in controller.last_event.metadata["objects"]
        if o["breakable"] is True
    ]
    all_breakable_objects = set(all_breakable_objects)
    for obj_type in all_breakable_objects:
        controller.step(action="MakeObjectsOfTypeUnbreakable", objectType=obj_type)


def remove_nan_inf_for_frames(frame: np.ndarray, type_of_frame: str = "") -> np.ndarray:
    is_nan = frame != frame
    is_inf = np.isinf(frame)
    if np.any(is_nan) or np.any(is_inf):
        mask = is_nan + is_inf
        get_logger().warning(f"Found nan {type_of_frame} {mask.sum()}")
        frame[mask] = 0
    return frame


def get_relative_stretch_current_arm_state(controller: Controller) -> dict:
    arm = controller.last_event.metadata["arm"]["joints"]
    z = arm[-1]["rootRelativePosition"]["z"]
    x = arm[-1]["rootRelativePosition"]["x"]
    assert abs(x - 0) < 1e-3
    y = arm[0]["rootRelativePosition"]["y"] - 0.16297650337219238  # TODO?
    return dict(x=x, y=y, z=z)


def position_distance(s1: dict, s2: dict, axis: tuple = ("x", "y", "z")) -> float:
    position1 = s1["position"] if isinstance(s1, dict) and "position" in s1 else s1
    position2 = s2["position"] if isinstance(s2, dict) and "position" in s2 else s2
    return (
        ("x" in axis) * (position1["x"] - position2["x"]) ** 2
        + ("y" in axis) * (position1["y"] - position2["y"]) ** 2
        + ("z" in axis) * (position1["z"] - position2["z"]) ** 2
    ) ** 0.5


def close_enough(
    current_obj_pose: dict,
    goal_obj_pose: dict,
    threshold: float = 0.2,
    norm: Literal[1, 2] = 1,
    axis: Union[tuple, str] = ("x", "y", "z"),
) -> bool:
    if isinstance(axis, str):
        axis = [axis]
    if norm == 1:
        current_obj_pose = (
            current_obj_pose["position"]
            if "position" in current_obj_pose
            else current_obj_pose
        )
        goal_obj_pose = (
            goal_obj_pose["position"] if "position" in goal_obj_pose else goal_obj_pose
        )
        position_close = [
            abs(current_obj_pose[k] - goal_obj_pose[k]) <= threshold for k in axis
        ]
        position_is_close = sum(position_close) == len(position_close)
    elif norm == 2:
        position_is_close = (
            position_distance(current_obj_pose, goal_obj_pose, axis=axis) <= threshold
        )
    else:
        raise NotImplementedError(norm)

    # rotation_close = [
    #     abs(current_obj_pose["rotation"][k] - goal_obj_pose["rotation"][k])
    #     <= threshold
    #     for k in ["x", "y", "z"]
    # ]
    # rotation_is_close = sum(rotation_close) == 3
    return position_is_close  # and rotation_is_close


def get_agent_teleport_kwargs_from_metadata(metadata: dict) -> dict:
    if "agent" in metadata:
        meta = metadata["agent"]
    else:
        assert metadata["name"] == "agent"
        meta = metadata
    location = {
        "x": meta["position"]["x"],
        "y": meta["position"]["y"],
        "z": meta["position"]["z"],
        "rotation": meta["rotation"]["y"],
        "horizon": meta["cameraHorizon"],
        "standing": meta["isStanding"],
    }
    return location


def object_ids_to_types(objects: list, object_ids: Union[list, str]) -> Optional[dict]:
    if isinstance(object_ids, str):
        object_ids = [object_ids]
    obj_types = {}
    for object_id in object_ids:
        for obj in objects:
            if obj["objectId"] == object_id:
                obj_types[object_id] = obj["objectType"]
                break
    if len(obj_types) != len(object_ids):
        get_logger().warning(
            f"query obj ids: {object_ids}, only found: {obj_types.keys()}"
        )
    return None


"""polar coordinates utils"""


def compute_pointgoal(
    source_position: np.ndarray,
    source_rotation: quaternion.quaternion,
    goal_position: np.ndarray,
    return_array: bool = True,
) -> Union[np.ndarray, tuple]:
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    rho, phi = cartesian_to_polar(direction_vector_agent[2], -direction_vector_agent[0])
    if return_array:
        return np.array([rho, phi], dtype=np.float32)
    else:
        return rho, phi


def quaternion_from_y_angle(angle: float) -> quaternion.quaternion:
    r"""Creates a quaternion from rotation angle around y axis."""
    return quaternion_from_coeff(
        np.array(
            [0.0, np.sin(np.pi * angle / 360.0), 0.0, np.cos(np.pi * angle / 360.0)]
        )
    )


def quaternion_from_coeff(coeffs: np.ndarray) -> quaternion.quaternion:
    r"""Creates a quaternions from coeffs in [x, y, z, w] format."""
    quat = np.quaternion(0, 0, 0, 0)
    quat.real = coeffs[3]
    quat.imag = coeffs[0:3]
    return quat


def cartesian_to_polar(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def quaternion_rotate_vector(quat, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion
    Args:
        quat: The quaternion to rotate by
        v: The vector to rotate
    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag


def get_polar_pos(euler_pos: dict, base_pos: dict, y_angle: float) -> tuple:
    if isinstance(y_angle, dict):
        y_angle = y_angle["y"]
    base_quaternion = quaternion_from_y_angle(y_angle)
    if "position" in euler_pos:
        euler_pos = euler_pos["position"]
    if "position" in base_pos:
        base_pos = base_pos["position"]
    rho, phi = compute_pointgoal(
        source_position=np.array([base_pos[k] for k in ["x", "y", "z"]]),
        source_rotation=base_quaternion,
        goal_position=np.array([euler_pos[k] for k in ["x", "y", "z"]]),
        return_array=False,
    )
    return rho, phi


"""Sampling Utils"""


def sample_obj_pos_over_space(
    objs_info: dict, space: dict, sampling_limit: int = 1000, raise_error: bool = False,
) -> Optional[dict]:
    # space should already consider the offset
    xlow, xhigh = space["low"]["x"], space["high"]["x"]
    zlow, zhigh = space["low"]["z"], space["high"]["z"]
    sample_objs = dict()
    sampled_bboxes = []  # (center, size) tuple
    for obj_id, obj_info in objs_info.items():
        obj_y = obj_info["position"]["y"]
        obj_size = obj_info["axisAlignedBoundingBox"]["size"]
        sample_pos = None  # placeholder
        success_sample = False
        for _ in range(sampling_limit):
            sample_x = random.uniform(xlow, xhigh)
            sample_z = random.uniform(zlow, zhigh)
            sample_pos = dict(x=sample_x, y=obj_y, z=sample_z)
            good_sample = True
            for bbox in sampled_bboxes:
                if bbox_intersect((sample_pos, obj_size), bbox):
                    good_sample = False
                    break
            if good_sample:
                sample_objs[obj_id] = sample_pos
                success_sample = True
                break
        if not success_sample:
            if raise_error:
                raise ValueError(f"reject sampling for {obj_id}")
            else:
                get_logger().warning(f"reject sampling for {obj_id}")
                return None
        sampled_bboxes.append((sample_pos, obj_size))
    return sample_objs


"""Arm Calculation Utils"""


def make_rotation_matrix(position, rotation):
    result = np.zeros((4, 4))
    r = R.from_euler("xyz", [rotation["x"], rotation["y"], rotation["z"]], degrees=True)
    result[:3, :3] = r.as_matrix()
    result[3, 3] = 1
    result[:3, 3] = [position["x"], position["y"], position["z"]]
    return result


def inverse_rot_trans_mat(mat):
    mat = np.linalg.inv(mat)
    return mat


def position_rotation_from_mat(matrix):
    result = {"position": None, "rotation": None}
    rotation = R.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
    rotation_dict = {
        "x": rotation[0] % 360,
        "y": rotation[1] % 360,
        "z": rotation[2] % 360,
    }
    result["rotation"] = rotation_dict
    position = matrix[:3, 3]
    result["position"] = {"x": position[0], "y": position[1], "z": position[2]}
    return result


def find_closest_inverse(deg):
    for k in saved_inverse_rotation_mats.keys():
        if abs(k - deg) < 5:
            return saved_inverse_rotation_mats[k]
    # if it reaches here it means it had not calculated the degree before
    rotation = R.from_euler("xyz", [0, deg, 0], degrees=True)
    result = rotation.as_matrix()
    inverse = inverse_rot_trans_mat(result)
    # print("WARNING: Had to calculate the matrix for ", deg)
    return inverse


def calc_inverse(deg):
    rotation = R.from_euler("xyz", [0, deg, 0], degrees=True)
    result = rotation.as_matrix()
    inverse = inverse_rot_trans_mat(result)
    return inverse


saved_inverse_rotation_mats = {i: calc_inverse(i) for i in range(0, 360, 45)}
saved_inverse_rotation_mats[360] = saved_inverse_rotation_mats[0]


def get_rel_obj1_to_obj2(obj1: dict, obj2: dict):
    """return relative position of obj1 in terms of obj2."""
    position = obj2["position"]
    rotation = obj2["rotation"]
    agent_translation = [position["x"], position["y"], position["z"]]
    # inverse_agent_rotation = inverse_rot_trans_mat(agent_rotation_matrix[:3, :3])
    assert abs(rotation["x"] - 0) < 0.01 and abs(rotation["z"] - 0) < 0.01
    inverse_agent_rotation = find_closest_inverse(rotation["y"])
    obj_matrix = make_rotation_matrix(obj1["position"], obj1["rotation"])
    obj_translation = np.matmul(
        inverse_agent_rotation, (obj_matrix[:3, 3] - agent_translation)
    )
    # add rotation later
    obj_matrix[:3, 3] = obj_translation
    result = position_rotation_from_mat(obj_matrix)
    return result


def get_abs_obj1_from_rel_obj1_to_obj2(rel_obj1: dict, obj2: dict) -> dict:
    position = obj2["position"]
    rotation = obj2["rotation"]
    agent_translation = [position["x"], position["y"], position["z"]]
    agent_rotation = R.from_euler(
        "xyz", [0, rotation["y"], 0], degrees=True
    ).as_matrix()
    obj_translation = (
        np.matmul(
            agent_rotation,
            [
                rel_obj1["position"]["x"],
                rel_obj1["position"]["y"],
                rel_obj1["position"]["z"],
            ],
        )
        + agent_translation
    )
    rel_obj_rotation = R.from_euler(
        "xyz",
        [
            rel_obj1["rotation"]["x"],
            rel_obj1["rotation"]["y"],
            rel_obj1["rotation"]["z"],
        ],
        degrees=True,
    )
    obj_rotation = agent_rotation.dot(rel_obj_rotation.as_rotvec())
    obj_rotation = R.from_rotvec(obj_rotation)
    result = {
        "position": {
            "x": obj_translation[0],
            "y": obj_translation[1],
            "z": obj_translation[2],
        },
        "rotation": {
            "x": obj_rotation.as_euler("xyz", degrees=True)[0] % 360,
            "y": obj_rotation.as_euler("xyz", degrees=True)[1] % 360,
            "z": obj_rotation.as_euler("xyz", degrees=True)[2] % 360,
        },
    }
    return result


# TODO this simplified version has bug for rotation
# def get_abs_obj1_from_rel_obj1_to_obj2(rel_obj1: dict, obj2: dict) -> dict:
#     agent_matrix = make_rotation_matrix(obj2["position"], obj2["rotation"])
#     rel_obj1_matrix = make_rotation_matrix(rel_obj1["position"], rel_obj1["rotation"])
#     obj_matrix = agent_matrix.dot(rel_obj1_matrix)
#     result = position_rotation_from_mat(obj_matrix)
#     return result


def get_abs_obj2_from_abs_obj1_and_rel_obj1_to_obj2(
    abs_obj1: dict, rel_obj1: dict
) -> dict:
    """metadata format."""
    position = abs_obj1["position"]
    rotation = abs_obj1["rotation"]
    agent_translation = np.array([position["x"], position["y"], position["z"]])
    agent_rotation = R.from_euler(
        "xyz", [rotation["x"], rotation["y"], rotation["z"]], degrees=True
    )
    obj_translation = np.matmul(
        agent_rotation.inv().as_matrix(),
        [
            rel_obj1["position"]["x"],
            rel_obj1["position"]["y"],
            rel_obj1["position"]["z"],
        ],
    )
    obj_translation *= -1
    obj_translation += agent_translation
    rel_obj_rotation = R.from_euler(
        "xyz",
        [
            rel_obj1["rotation"]["x"],
            rel_obj1["rotation"]["y"],
            rel_obj1["rotation"]["z"],
        ],
        degrees=True,
    )
    obj_rotation = agent_rotation * rel_obj_rotation
    result = {
        "position": {
            "x": obj_translation[0],
            "y": obj_translation[1],
            "z": obj_translation[2],
        },
        "rotation": {
            "x": obj_rotation.as_euler("xyz", degrees=True)[0],
            "y": obj_rotation.as_euler("xyz", degrees=True)[1],
            "z": obj_rotation.as_euler("xyz", degrees=True)[2],
        },
    }
    return result


# hacky objects that move
CONSTANTLY_MOVING_OBJECTS = {
    "FloorPlan1": {"Egg|-02.01|+00.81|+01.25"},
    "FloorPlan2": {"Egg|+00.06|+00.97|-00.17"},
    "FloorPlan4": {"Egg|-03.32|+01.31|+02.85"},
    "FloorPlan5": {"Egg|-00.14|+00.78|-01.92"},
    "FloorPlan6": {"Egg|-02.53|+00.60|-00.71"},
    "FloorPlan10": {"Egg|+00.89|+01.16|+01.09"},
    "FloorPlan11": {"Egg|-02.32|+00.80|-01.72"},
    "FloorPlan15": {"Tomato|-02.30|+00.97|+03.69"},
    "FloorPlan16": {"Ladle|+02.61|+01.04|-01.50"},
    "FloorPlan20": {"Vase|+01.50|+00.56|+02.45"},
    "FloorPlan21": {"Lettuce|-00.28|+00.97|+01.13"},
    "FloorPlan22": {"Apple|+00.28|+01.15|+01.58"},
    "FloorPlan23": {"Egg|-00.46|+01.40|-01.01"},
    "FloorPlan24": {"Microwave|-01.53|+01.25|+03.88"},
    "FloorPlan27": {"Ladle|-00.10|+00.95|+02.55"},
    "FloorPlan28": {"Apple|-00.44|+01.00|-01.48"},
    "FloorPlan217": {"Chair|-04.74|+00.01|+04.61"},
    "FloorPlan229": {"Box|-03.42|+00.59|+02.44"},
    "FloorPlan316": {"Pen|+00.14|+00.70|-02.21", "Pencil|+00.21|+00.70|-02.20"},
    "FloorPlan326": {"BaseballBat|-02.90|+00.06|-02.70"},
    "FloorPlan416": {"ToiletPaper|-01.57|+00.64|+00.05"},
    "FloorPlan418": {"ToiletPaper|-00.37|+00.05|-03.86"},
}
