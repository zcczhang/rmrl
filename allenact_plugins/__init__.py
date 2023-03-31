try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    from allenact_plugins._version import __version__
except ModuleNotFoundError:
    __version__ = None

# setting mujoco path variable
import os

os.environ[
    "LD_LIBRARY_PATH"
] = f":{os.environ['HOME']}/.mujoco/mujoco210/bin:/usr/lib/nvidia"
