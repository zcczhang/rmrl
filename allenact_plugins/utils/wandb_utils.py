import copy
from typing import Union

import gym.spaces
import numpy as np

from .array_tensor_utils import any_to_primitive

# from allenact_plugins.stretch_manipulathor_plugin.dataset_utils import TaskData


__all__ = [
    "thor_points_to_wandb_points",
    "sphere_to_corner_points",
    "center_size_to_corner_points",
    "space_to_corner_points",
]


def thor_points_to_wandb_points(
    points: Union[list, np.ndarray], return_list: bool = True
):
    """(x, y, z, ...) in thor to (z, -x, y, ...)"""
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    assert points.ndim == 2, (points.ndim, points.shape)
    xyz_shape = points.shape[1]
    transpose_shape = [2, 0, 1] + [3 + i for i in range(xyz_shape - 3)]
    flipped_points = points[:, transpose_shape]
    flipped_points[:, 0] = -flipped_points[:, 0]
    return any_to_primitive(flipped_points) if return_list else flipped_points


def sphere_to_corner_points(center: dict, r: float) -> list:
    """corner points from sphere of dict of xyz and radius."""
    x, y, z = center["x"], center["y"], center["z"]
    return [
        [x - r, y - r, z - r],
        [x - r, y - r, z + r],
        [x - r, y + r, z - r],
        [x - r, y + r, z + r],
        [x + r, y - r, z - r],
        [x + r, y - r, z + r],
        [x + r, y + r, z - r],
        [x + r, y + r, z + r],
    ]


def center_size_to_corner_points(center: dict, size: dict) -> list:
    cx, cy, cz = center["x"], center["y"], center["z"]
    hx, hy, hz = size["x"] / 2, size["y"] / 2, size["z"] / 2
    return [
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ]


def space_to_corner_points(
    space: Union[gym.spaces.Box, "TaskData.Space", dict]
) -> list:
    from allenact_plugins.stretch_manipulathor_plugin.dataset_utils import (
        TaskData,
    )  # circular import issue

    assert isinstance(space, (gym.spaces.Box, TaskData.Space, dict)), type(space)
    if isinstance(space, dict):
        space = TaskData.Space(copy.deepcopy(space))
    l, h = space.low, space.high
    lx, ly, lz = l if isinstance(space, gym.spaces.Box) else l.x, l.y, l.z
    hx, hy, hz = h if isinstance(space, gym.spaces.Box) else h.x, h.y, h.z
    return [
        [lx, ly, lz],
        [lx, hy, lz],
        [lx, ly, hz],
        [lx, hy, hz],
        [hx, ly, lz],
        [hx, hy, lz],
        [hx, ly, hz],
        [hx, hy, hz],
    ]
