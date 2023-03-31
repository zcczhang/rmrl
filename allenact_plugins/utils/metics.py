from typing import List, Union, Optional, Dict, Sequence, Tuple

import gym
import numpy as np
from npeet.entropy_estimators import entropy as npeet_entropy
from npeet.entropy_estimators import entropyd as npeet_entropyd
from scipy.stats import entropy as scipy_entropy

from allenact.utils.system import get_logger

__all__ = [
    "states_measure",
    "states_entropy",
    "calculate_entropy_from_occupancy_grid",
    "spl_metric",
    "np_str2pooling",
    "arc_length",
    "stable_mul",
    "bbox_intersect",
    "is_bbox_inside",
    "assert_eq",
]


def states_measure(
    xyzs: Union[List[np.ndarray], np.ndarray],
    metrics: str,
    mean_xyz: bool = True,
    normalize_first: bool = False,
) -> Union[np.ndarray, float]:
    """
    Args:
        xyzs: shape L, d (e.g. d=3 for xyz poses)
        metrics: ["min", "max", "std", "mean", "median"] metrics for measure (e.g. std of history)
        mean_xyz: whether mean xyz after std
    """
    if isinstance(xyzs, list):
        xyzs = np.stack(xyzs)
    if normalize_first:
        xyzs = (xyzs - np.mean(xyzs, axis=0)) / np.std(xyzs, axis=0)
        xyzs = np.nan_to_num(xyzs, nan=0.0)
    measure = np_str2pooling(metrics)(xyzs, axis=0)
    return np.mean(measure) if mean_xyz else measure


def calculate_entropy_from_occupancy_grid(occupancy_grid: np.ndarray) -> float:
    # Reshape the occupancy grid into a 1D array
    flatten_grid = occupancy_grid.reshape(-1)
    # Normalize the count of each grid cell to obtain the probability distribution
    prob_dist = flatten_grid / np.sum(flatten_grid)
    # Calculate the entropy using the scipy entropy function
    entropy = scipy_entropy(prob_dist, base=2)
    return entropy


def states_entropy(
    state_history: Union[np.ndarray, List[np.ndarray]],
    *,
    entropy_fn: str = "scipy",
    grid_size: Optional[Union[int, List[int]]],
    world_space: Optional[gym.spaces.Box],
) -> float:
    if not isinstance(state_history, np.ndarray):
        state_history = np.stack(state_history, axis=0)
    if len(state_history) < 10:
        return -1.0
    if entropy_fn == "npeet":  # continuous
        return npeet_entropy(state_history)
    elif entropy_fn == "npeetd":  # discrete
        return npeet_entropyd(state_history)
    state_dim = state_history.shape[1]
    assert state_dim % 3 == 0, f"{state_dim} has to be multiple of 3 (xyz formats)"
    # Define the number of bins in each dimension
    n_xyzs = state_dim // 3
    nbins = (
        [grid_size] * state_dim
        if isinstance(grid_size, (int, float))
        else grid_size * n_xyzs
    )
    # Create the empty occupancy grid with the same number of bins in each dimension
    occupancy_grid = np.zeros(nbins)

    state_low = np.concatenate([world_space.low for _ in range(n_xyzs)])
    state_high = np.concatenate([world_space.high for _ in range(n_xyzs)])

    # Discretize the state_history into the occupancy_grid
    for i in range(state_history.shape[0]):
        state = state_history[i]
        # normalize the state
        normalized_state = np.clip(
            (state - state_low) / (state_high - state_low), 0, 1 - 1e-7
        )
        # Find the bin for each dimension of the state
        bin_indices = [
            int(np.floor(normalized_state[j] * nbins[j])) for j in range(state_dim)
        ]
        # Increment the corresponding bin in the occupancy grid
        occupancy_grid[tuple(bin_indices)] += 1
    return calculate_entropy_from_occupancy_grid(occupancy_grid)


def spl_metric(
    success: bool, optimal_distance: float, travelled_distance: float
) -> Optional[float]:
    if not success:
        return 0.0
    elif optimal_distance < 0:
        return None
    elif optimal_distance == 0:
        if travelled_distance == 0:
            return 1.0
        else:
            return 0.0
    else:
        travelled_distance = max(travelled_distance, optimal_distance)
        return optimal_distance / travelled_distance


def np_str2pooling(pool: str):
    if pool in ["min", "max"]:
        pool = "a" + pool
    for fn in [np.mean, np.std, np.max, np.min, np.median]:
        if fn.__name__ == pool:
            return fn
    raise ValueError(f"pooling method {pool} is not available")


def arc_length(d: float, r: float = 0.385) -> float:
    """calculate the arc length based on distance between two points `d` and
    radius of the circle `r`, by `arc = 2r sin^{-1}(d/2r)`.

    Used in Sawyer Door environment
    """
    return 2.0 * r * np.arcsin(d / (2 * r))


def stable_mul(value: float, coefficient: float, threshold: float = 1e-7) -> float:
    """multiply a small value with large coefficient, and return 0.0 if the
    result is smaller than the tolerance threshold."""
    return 0.0 if abs(value * coefficient) < threshold else value * coefficient


def bbox_intersect(bbox1: Tuple[dict, dict], bbox2: Tuple[dict, dict]):
    b1_center, b1_size = bbox1
    b2_center, b2_size = bbox2

    b1_min = [b1_center[i] - b1_size[i] / 2 for i in ["x", "y", "z"]]
    b1_max = [b1_center[i] + b1_size[i] / 2 for i in ["x", "y", "z"]]
    b2_min = [b2_center[i] - b2_size[i] / 2 for i in ["x", "y", "z"]]
    b2_max = [b2_center[i] + b2_size[i] / 2 for i in ["x", "y", "z"]]

    # intersect_min = [max(b1_min[i], b2_min[i]) for i in range(3)]
    # intersect_max = [min(b1_max[i], b2_max[i]) for i in range(3)]
    # intersect_size = [intersect_max[i] - intersect_min[i] for i in range(3)]
    # intersect_size = [s if s > 0 else 0 for s in intersect_size]
    # intersect_volume = intersect_size[0] * intersect_size[1] * intersect_size[2]
    #
    # b1_volume = b1_size["x"] * b1_size["y"] * b1_size["z"]
    # b2_volume = b2_size["x"] * b2_size["y"] * b2_size["z"]
    #
    # IoU = intersect_volume / (b1_volume + b2_volume - intersect_volume)

    if (
        b1_max[0] < b2_min[0]
        or b1_min[0] > b2_max[0]
        or b1_max[1] < b2_min[1]
        or b1_min[1] > b2_max[1]
        or b1_max[2] < b2_min[2]
        or b1_min[2] > b2_max[2]
    ):
        return False
    return True


def is_bbox_inside(bbox1: Tuple[dict, dict], bbox2: Tuple[dict, dict], threshold=0.0):
    """if bbox1 is inside the bbox2 within a small threshold."""
    b1_center, b1_size = bbox1
    b2_center, b2_size = bbox2

    b1_min = [b1_center[i] - b1_size[i] / 2 for i in ["x", "y", "z"]]
    b1_max = [b1_center[i] + b1_size[i] / 2 for i in ["x", "y", "z"]]
    b2_min = [b2_center[i] - b2_size[i] / 2 for i in ["x", "y", "z"]]
    b2_max = [b2_center[i] + b2_size[i] / 2 for i in ["x", "y", "z"]]

    if (
        # xz should be inside
        b1_max[0] > b2_max[0] + threshold
        or b1_min[0] < b2_min[0] - threshold
        or b1_max[1] > b2_max[1] + threshold
        or b1_min[1] < b2_min[1] - threshold
        # bbox1 min height should be smaller than bbox2 max height as inside
        # TODO tune?
        or b1_min[2]
        > b2_max[2] - threshold / 2  # `- threshold` in case flat container (e.g. plate)
    ):
        return False
    return True


def assert_eq(
    a: Union[str, float, Dict[str, float]],
    b: Union[str, float, Dict[str, float]],
    tolerance: float = 1e-7,
    reason: Optional[str] = None,
    echo: bool = True,
):
    reason_str = (f"{reason} " if reason else "") + "assertion failed: "
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys():
            assert_eq(a[k], b[k], tolerance, reason, echo)
    else:
        if isinstance(a, str) and isinstance(b, str):
            assert a == b, reason_str + f"{a} != {b}"
        elif isinstance(a, Sequence) and isinstance(b, Sequence):
            assert_eq(len(a), len(b), tolerance, reason, echo)
            for i in range(len(a)):
                assert_eq(a[i], b[i], tolerance, reason)
        elif a != b:
            diff = abs(a - b)
            assert diff < tolerance, (
                reason_str
                + f"abs({a} - {b}) > {diff} not within the tolerance {tolerance}"
            )
            if echo:
                get_logger().debug(
                    (f"{reason} assertion check: " if reason else "")
                    + f"slight diff {diff} within threshold {tolerance}"
                )
