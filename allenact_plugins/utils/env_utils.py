import os

import gym
import numpy as np

__all__ = [
    "register_gym_env",
    "get_file",
    "dict_space_to_gym_space",
    "dict_space_to_gym_Box_space",
]


def register_gym_env(env_id: str, **kwargs):
    """decorator for gym env registration."""

    def _register(cls):
        gym.register(
            id=env_id, entry_point=f"{cls.__module__}:{cls.__name__}", kwargs=kwargs
        )
        return cls

    return _register


def get_file(f_cur: str, tar_dir: str, tar_name: str):
    """return abs path of the target file in terms of current file.

    Example: get f_name file abs path in current file.
        - parent_dir
             - tar_dir
                - tar_name
            - f_cur
        or
        - parent_dir
            - tar_dir
                - tar_name
            - cur_dir
                f_cur
    """
    parent_dir = os.path.dirname(os.path.realpath(f_cur))
    tar_dir_path = os.path.join(parent_dir, tar_dir)
    if not os.path.exists(tar_dir_path):
        tar_dir_path = os.path.join(os.path.dirname(parent_dir), tar_dir)
    return os.path.join(tar_dir_path, tar_name)


def dict_space_to_gym_space(space) -> gym.spaces.Dict:
    _space = space if isinstance(space, dict) else space.to_dict()
    return gym.spaces.Dict(
        {
            i: gym.spaces.Box(
                low=_space["low"][i],
                high=_space["high"][i],
                shape=(1,),
                dtype=np.float32,
            )
            for i in ["x", "y", "z"]
        }
    )


def dict_space_to_gym_Box_space(space) -> gym.spaces.Box:
    _space = space if isinstance(space, dict) else space.to_dict()
    return gym.spaces.Box(
        low=np.array([_space["low"][i] for i in ["x", "y", "z"]]),
        high=np.array([_space["high"][i] for i in ["x", "y", "z"]]),
        shape=(3,),
        dtype=np.float32,
    )
