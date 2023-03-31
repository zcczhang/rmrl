import re
import shutil
from typing import Optional, Union

import allenact_plugins.utils as U
from allenact.utils.misc_utils import deprecated
from allenact_plugins.utils import round_metrics, assert_eq
from allenact_plugins.utils.config_utils import Component
from .stretch_arm_environment import StretchManipulaTHOREnvironment
from .stretch_utils import get_agent_teleport_kwargs_from_metadata

__all__ = [
    "TaskData",
    "check_task_with_env",
    "parse_obj_data_from_info",
    "parse_task_data",
    "parse_dataset",
    "filter_floor_plan",
]


class TaskData(Component):
    """just a helper class make data elements accessible as attributes."""

    class XYZ(Component):
        def __init__(self, xyz: Union[dict, "TaskData.XYZ"]):
            super().__init__()
            if isinstance(xyz, dict):
                self.x, self.y, self.z = xyz["x"], xyz["y"], xyz["z"]
            else:
                self.x, self.y, self.z = xyz.x, xyz.y, xyz.z

    class Obj(Component):
        def __init__(
            self, data: Union[dict, "TaskData.Obj"], dummy_target: bool = False
        ):
            super().__init__()
            if isinstance(data, dict):
                if dummy_target:
                    self.object_location = self.center = TaskData.XYZ(
                        data["object_location"]
                    )
                    self.object_name = "RandomTarget"
                    self.size = TaskData.XYZ(dict(x=0.05, y=0.05, z=0.05))
                else:
                    self.from_dict(**data)
            else:
                self.object_location = TaskData.XYZ(data.object_location)
                self.center = TaskData.XYZ(data.center)
                self.size = TaskData.XYZ(data.size)
                if not dummy_target:
                    self.object_id = data.object_id
                    self.object_name = data.object_name
                    self.object_rotation = TaskData.XYZ(data.object_rotation)
                    self.countertop_id = data.countertop_id
                    self.corner_points = data.corner_points
                else:
                    self.object_name = "RandomTarget"

        def from_dict(
            self,
            object_id: Optional[str],
            object_name: str,
            object_location: Union[dict, "TaskData.XYZ"],
            object_rotation: Union[dict, "TaskData.XYZ"],
            countertop_id: str,
            axisAlignedBoundingBox: Optional[dict] = None,
            **kwargs,
        ):
            self.object_id = object_id
            self.object_name = object_name
            self.object_location = TaskData.XYZ(object_location)
            self.object_rotation = TaskData.XYZ(object_rotation)
            self.countertop_id = countertop_id
            if axisAlignedBoundingBox is None:
                # may initialize from another TaskData
                self.corner_points = kwargs.pop("corner_points")
                self.center = TaskData.XYZ(kwargs.pop("center"))
                self.size = TaskData.XYZ(kwargs.pop("size"))
                assert len(kwargs) == 0, kwargs
            else:
                assert len(kwargs) == 0, kwargs
                self.corner_points: list = axisAlignedBoundingBox["cornerPoints"]
                # could be different from object id
                self.center = TaskData.XYZ(axisAlignedBoundingBox["center"])
                self.size = TaskData.XYZ(axisAlignedBoundingBox["size"])

    class Agent(Component):
        def __init__(self, data: Union[dict, "TaskData.Agent"]):
            super().__init__()
            if isinstance(data, dict):
                self.from_dict(**data)
            else:
                self.name = data.name
                self.position = TaskData.XYZ(data.position)
                self.rotation = TaskData.XYZ(data.rotation)
                self.cameraHorizon = data.cameraHorizon
                self.isStanding = data.isStanding

        def to_dict(self, teleport_format: bool = False) -> dict:
            return (
                super().to_dict()
                if not teleport_format
                else get_agent_teleport_kwargs_from_metadata(self.to_dict(False))
            )

        def from_dict(
            self,
            *,
            name: str = "agent",
            position: Union[dict, "TaskData.XYZ"],
            rotation: Union[dict, "TaskData.XYZ"],
            cameraHorizon: Union[int, float],
            isStanding: bool,
            **kwargs,
        ):
            kwargs.pop("inHighFrictionArea", None)
            assert len(kwargs) == 0, kwargs
            self.name = name
            self.position = TaskData.XYZ(position)
            self.rotation = TaskData.XYZ(rotation)
            self.cameraHorizon = cameraHorizon
            self.isStanding = isStanding

    class Space(Component):
        def __init__(self, data: Optional[Union[dict, "TaskData.Space"]]):
            super().__init__()
            if data is None:
                self.low, self.high = None, None
            else:
                if isinstance(data, dict):
                    self.low, self.high = (
                        TaskData.XYZ(data["low"]),
                        TaskData.XYZ(data["high"]),
                    )
                else:
                    self.low, self.high = (
                        TaskData.XYZ(data.low),
                        TaskData.XYZ(data.high),
                    )

    def __init__(
        self,
        *,
        scene_name: str = "FloorPlan_ExpRoom",
        initial_agent_pos: Union[dict, "TaskData.Agent"],
        init_object: Union[dict, "TaskData.Obj"],
        goal_object: Union[dict, "TaskData.Obj"],
        random_sample_space: Optional[dict] = None,
        task_name: Optional[str] = None,
        is_obj_goal: bool,
    ):
        super().__init__()
        self.scene_name = scene_name
        self.initial_agent_pos = TaskData.Agent(initial_agent_pos)
        self.init_object = TaskData.Obj(init_object)
        self.goal_object = TaskData.Obj(goal_object, dummy_target=not is_obj_goal)
        self.random_sample_space = TaskData.Space(random_sample_space)
        self.is_obj_goal = is_obj_goal
        self.task_name = task_name or self.parse_task_name()
        self.lock()

    @staticmethod
    def parse_object_name(obj_name: str):
        return obj_name.replace("_", " ").lower()

    def parse_task_name(self):
        source_obj_name = self.parse_object_name(self.init_object.object_name)
        return (
            f"Put {source_obj_name} "
            f"into {self.parse_object_name(self.goal_object.object_name)}."
            if self.is_obj_goal
            else f"Put {source_obj_name} to "
            f"{round_metrics(tuple(self.goal_object.object_location.to_dict().values()), n=4)}."
        )


def check_task_with_env(env: StretchManipulaTHOREnvironment, task_data: TaskData):
    def _check_obj(obj_id: str, task_obj: TaskData.Obj):
        env_obj_info = env.get_object_by_id(obj_id)
        name = task_obj.object_name
        assert_eq(name, env_obj_info["name"], reason=f"{name} name")
        assert_eq(
            task_obj.object_location.to_dict(),
            env_obj_info["position"],
            reason=f"{name} pos",
        )
        assert_eq(
            task_obj.object_rotation.to_dict(),
            env_obj_info["rotation"],
            reason=f"{name} rot",
        )
        parent_receptacles = env_obj_info["parentReceptacles"]
        assert_eq(
            len(env_obj_info["parentReceptacles"]),
            1,
            reason=f"{name} only 1 parent receptacle",
        )
        assert_eq(
            task_obj.countertop_id,
            parent_receptacles[0],
            reason=f"{name} parent receptacle id",
        )
        assert_eq(
            task_obj.corner_points,
            env_obj_info["axisAlignedBoundingBox"]["cornerPoints"],
            reason=f"{name} axisAlignedBoundingBox corner points",
        )
        assert_eq(
            task_obj.center.to_dict(),
            env_obj_info["axisAlignedBoundingBox"]["center"],
            reason=f"{name} center",
        )
        assert_eq(
            task_obj.size.to_dict(),
            env_obj_info["axisAlignedBoundingBox"]["size"],
            reason=f"{name} size",
        )

    def _check_agent(task_agent: TaskData.Agent):
        agent_info = env.last_event.metadata["agent"]
        assert_eq(task_agent.name, agent_info["name"], reason="agent name")
        assert_eq(
            task_agent.position.to_dict(), agent_info["position"], reason="agent pos",
        )
        assert_eq(
            task_agent.rotation.to_dict(), agent_info["rotation"], reason="agent rot",
        )
        assert_eq(
            task_agent.cameraHorizon,
            agent_info["cameraHorizon"],
            reason="agent camera horizon",
        )
        assert_eq(
            task_agent.isStanding, agent_info["isStanding"], reason="agent is standing",
        )

    _check_agent(task_data.initial_agent_pos)
    _check_obj(task_data.init_object.object_id, task_data.init_object)
    if task_data.is_obj_goal:
        _check_obj(task_data.goal_object.object_id, task_data.goal_object)


def parse_obj_data_from_info(obj_info: dict, countertop_id: Optional[str]) -> dict:
    countertop_id = countertop_id or obj_info["parentReceptacles"][0]
    return dict(
        object_id=obj_info["objectId"],
        object_name=obj_info["name"],
        object_location=obj_info["position"],
        object_rotation=obj_info["rotation"],
        countertop_id=countertop_id,
        axisAlignedBoundingBox=obj_info["axisAlignedBoundingBox"],
    )


@deprecated
def parse_task_data(task_data: dict, data_path: Optional[str] = None) -> dict:
    """make useful info as a separate un-nested dict for updating task_info."""
    data_path = data_path or task_data.get("data_path", None)
    init_obj_info = task_data["init_object"]
    goal_obj_info = task_data["goal_object"]
    task_info = dict(
        scene_name=task_data["scene_name"],
        initial_agent_pos=task_data["initial_agent_pose"],
        source_object_id=init_obj_info["object_id"],
        init_obj_location=init_obj_info["object_location"],
        initial_obj_countertop_id=init_obj_info["countertop_id"],
        # init_obj_agent_pose=init_obj_info["agent_pose"],
        goal_object_id=goal_obj_info["object_id"],
        goal_obj_location=goal_obj_info["object_location"],
        goal_obj_countertop_id=goal_obj_info["countertop_id"],
        # goal_obj_agent_pose=goal_obj_info["agent_pose"],
    )
    if data_path is not None:
        scene, obj, goal = parse_dataset(data_path, remove_physics=True)
        task_info["task_name"] = f"Bring {obj} to {goal} in {scene}"
    return task_info


@deprecated
def parse_dataset(file: str, remove_physics: bool = False):
    """could use for get obj, goal, and scene info, or as a key for sort()"""
    file = file.split("/")[-1]
    assert file.endswith(".json")
    match = re.search(r"tasks_obj_(.*)_to_(.*)_scene_(.*)_physics.json", file)
    if match:
        obj = match.group(1)
        goal = match.group(2)
        scene = match.group(3)
        if remove_physics:
            scene = scene.split("_physics")[0]
        return scene, obj, goal
    else:
        raise ValueError(file, match)


@deprecated
def filter_floor_plan(src_dir: str, dst_dir: str):
    pattern = re.compile("tasks_obj_.*_scene_(.*)_physics.json")
    for filename in U.f_listdir(src_dir):
        match = pattern.match(filename)
        if match:
            floor_plan = match.group(1)
            new_dst_dir = U.f_join(dst_dir, floor_plan)
            U.f_mkdir(new_dst_dir)
            src_path = U.f_join(src_dir, filename)
            dst_path = U.f_join(new_dst_dir, filename)
            shutil.copy(src_path, dst_path)
