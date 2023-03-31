import datetime
from typing import Any, Type, Union, Dict
from typing import Optional
from typing import Tuple, List

import gym
import numpy as np

import allenact_plugins.utils as U
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, SubTaskType
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.system import get_logger
from allenact_plugins.stretch_manipulathor_plugin.stretch_tasks import (
    AbstractStretchTask,
)
from allenact_plugins.utils import (
    add_boundary_from_success,
    thor_points_to_wandb_points,
)
from allenact_plugins.wrappers import EnvWrapperType

__all__ = [
    "TrainingVideoCallbackSensor",
    "CloudPointCallbackSensor",
    "ThorTrainingVideoCallbackSensor",
    "ThorPointCloudCallbackSensor",
]


class TrainingVideoCallbackSensor(
    Sensor[Type[EnvWrapperType], Task[Type[EnvWrapperType]]]
):
    def __init__(self, uuid: str = "video_callback", **kwargs: Any):
        observation_space = gym.spaces.Space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.logged_video_files = []

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[str], Optional[List[np.ndarray]]]:
        un_sync_vid_file = None
        un_sync_vid_frames = None
        if hasattr(env, "un_sync_video_file"):
            un_sync_vid_file = env.un_sync_video_file
            un_sync_vid_frames = env.un_sync_video_frames
            if un_sync_vid_frames is not None:
                # L, C, H, W in wandb video
                un_sync_vid_frames = np.array(un_sync_vid_frames).transpose(
                    [0, 3, 1, 2]
                )
            env.un_sync_video_file = None
            env.un_sync_video_frames = None
        return un_sync_vid_file, un_sync_vid_frames


class CloudPointCallbackSensor(
    Sensor[Type[EnvWrapperType], Task[Type[EnvWrapperType]]]
):
    def __init__(
        self, uuid: str = "point_cloud_callback", mode: str = "train", **kwargs: Any,
    ):
        observation_space = gym.spaces.Space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.mode = mode

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[SubTaskType],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ):
        if (
            hasattr(env, "log_point_cloud")
            and hasattr(env, "un_sync_video_file")
            and (env.log_point_cloud is True or self.mode != "train")
        ):
            cloud_points = env.points
            env.start_new_recording_point_cloud()
            box = []
            if hasattr(env, "random_target_space"):
                box.append(self.get_space_corners(env.random_target_space, [0, 0, 255]))
            # if hasattr(env, "reversible_space"):
            #     box.append(self.get_space_corners(env.reversible_space, [255, 255, 0]))
            if hasattr(env, "goal_space"):
                box.append(self.get_space_corners(env.goal_space, [255, 255, 0]))
            if hasattr(env, "table_space"):
                box.append(self.get_space_corners(env.table_space, [0, 255, 255]))
            return cloud_points, box
        else:
            return None

    @staticmethod
    def get_space_corners(
        space: gym.spaces.Box, color: Union[list, str]
    ) -> Dict[str, list]:
        box = {}
        l, h = space.low, space.high
        lx, ly, lz = l
        hx, hy, hz = h
        box["corners"] = [
            [lx, ly, lz],
            [lx, hy, lz],
            [lx, ly, hz],
            [lx, hy, hz],
            [hx, ly, lz],
            [hx, hy, lz],
            [hx, ly, hz],
            [hx, hy, hz],
        ]
        box["color"] = color
        return box


class ThorTrainingVideoCallbackSensor(
    Sensor[Type[EnvWrapperType], Task[Type[EnvWrapperType]]]
):
    def __init__(
        self,
        uuid: str = "thor_video_callback",
        mode: str = "debug",
        save: bool = False,
        save_path: str = ".",
        record_rule: Tuple[int, int, int] = (125000, 125000, 6),
        record_start: int = 0,
        record_interval: int = 1e3,
        tasks: int = 1e3,
        fps: int = 30,
        **kwargs,
    ):
        observation_space = gym.spaces.Space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.mode = mode
        self.save = save
        # generate file name for callback even if not saving locally to the folder
        self.save_path = U.f_join(
            save_path, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"),
        )  # differentiate between different processes
        if save:
            U.f_mkdir(self.save_path)

        _, _, self._num_record_tasks = record_rule
        self.vid_writer = U.VideoTensorWriter(folder=self.save_path, fps=fps)
        self.already_recorded_tasks = 0

        self.un_sync_video_file = None
        self.un_sync_video_frames = None

    def get_observation(
        self,
        env: Type[EnvWrapperType],
        task: Optional[AbstractStretchTask],
        *args,
        gym_obs: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[str], Optional[List[np.ndarray]]]:
        """calling once task is_done()"""
        self.add_task_frames(task)

        un_sync_vid_file = None
        un_sync_vid_frames = None
        if self.un_sync_video_frames is not None:
            if len(self.un_sync_video_frames) > 0:
                # L, C, H, W in wandb video
                un_sync_vid_frames = np.array(self.un_sync_video_frames).transpose(
                    [0, 3, 1, 2]
                )
            un_sync_vid_file = self.un_sync_video_file
            self.un_sync_video_file = None
            self.un_sync_video_frames = None
        return un_sync_vid_file, un_sync_vid_frames

    def add_task_frames(self, task: AbstractStretchTask):
        """flip task.record in task sampler already."""
        if not task.record:
            return
        for idx, frame in enumerate(task.frame_histories):
            success = idx == len(task.frame_histories) - 1 and task.success
            self.vid_writer.add_frame(add_boundary_from_success(frame, success=success))
        self.already_recorded_tasks += 1
        if self.already_recorded_tasks >= self._num_record_tasks:
            get_logger().debug(f"record ended at {task.env.cum_env_steps} steps")
            self.un_sync_video_file = self.vid_writer.save(
                step=task.env.cum_env_steps, suffix=task.process_ind, save=self.save
            )
            self.already_recorded_tasks = 0
            self.un_sync_video_frames = self.vid_writer.frames
            self.vid_writer.clear()


class ThorPointCloudCallbackSensor(
    Sensor[Type[EnvWrapperType], Task[Type[EnvWrapperType]]]
):
    def __init__(
        self, uuid: str = "thor_video_callback", num_steps_recording: int = 1e5
    ):
        observation_space = gym.spaces.Space()
        super().__init__(**prepare_locals_for_super(locals()))
        self.num_steps_recording = num_steps_recording
        self.obj_histories = []
        self.container_corners = []
        self.container_names = []

    def get_observation(
        self,
        env: EnvWrapperType,
        task: Optional[AbstractStretchTask],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if not task.record_during_training:
            return None
        if not hasattr(task, "state_history") or task.state_history is None:
            get_logger().warning(
                f"not stored state history for logging point cloud: {getattr(task, 'state_history', 'no attr')}"
            )
            return None
        obj_history = task.state_history["obj"]
        point_color = [0, 255, 0] if task.success else [255, 0, 0]
        self.obj_histories.extend([obj + point_color for obj in obj_history])
        if task.is_obj_goal:
            self.container_corners.append(
                task.current_state_metadata["obj_goal_state"][
                    "objectOrientedBoundingBox"
                ]["cornerPoints"]
            )
            self.container_names.append(task.task_data.goal_object.object_name)
        point_cloud = None
        if len(self.obj_histories) >= self.num_steps_recording:
            boxes = []
            table = task.env.get_object_by_id(task.task_data.init_object.countertop_id)[
                "axisAlignedBoundingBox"
            ]["cornerPoints"]
            boxes.append(
                dict(
                    corners=thor_points_to_wandb_points(table),
                    color=[0, 255, 255],
                    lable="CounterTop",
                )
            )
            for container, name in zip(self.container_corners, self.container_names):
                boxes.append(
                    dict(
                        corners=thor_points_to_wandb_points(container),
                        color=[255, 255, 255],
                        lable=name,
                    )
                )
            point_cloud = dict(
                type="lidar/beta",
                points=thor_points_to_wandb_points(
                    self.obj_histories, return_list=False
                ),
                boxes=np.array(boxes),
            )
            self.obj_histories = []
            self.container_corners = []
            self.container_names = []
        return point_cloud
