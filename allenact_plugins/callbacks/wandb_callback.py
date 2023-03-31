import os
from collections import defaultdict
from typing import List, Dict, Any, Sequence, Optional

import numpy as np
import wandb
from matplotlib.figure import Figure

from allenact.base_abstractions.callbacks import Callback
from allenact.base_abstractions.sensor import Sensor
from allenact_plugins.callbacks.callback_sensors import (
    TrainingVideoCallbackSensor,
    CloudPointCallbackSensor,
    ThorTrainingVideoCallbackSensor,
    ThorPointCloudCallbackSensor,
)
from allenact_plugins.utils import Config, f_mkdir
from allenact_plugins.wrappers import ExperimentConfigBase

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class WBCallback(Callback):
    run_name: str
    cfg: Config
    mode: Literal["train", "valid", "test"]

    def __init__(self):
        super().__init__()
        self.saved_max_num_hard_reset_metrics = defaultdict(lambda: 0)
        self.num_processes = None

    def setup(
        self,
        name: str,
        config: ExperimentConfigBase,
        mode: Literal["train", "valid", "test"],
        **kwargs,
    ) -> None:
        """Called once before training begins."""
        self.run_name = config.tag()
        self.cfg = config.cfg
        self.mode = mode
        callback_kwargs = config.cfg.callback_kwargs
        os.environ["WANDB_MODE"] = "online" if callback_kwargs["sync"] else "offline"
        wandb_init_kwargs = dict(
            project=callback_kwargs["wandb_project"],
            entity=callback_kwargs["wandb_entity"],
            name=name,
            config=kwargs,
        )
        if callback_kwargs.get("output_dir", None):
            wandb_init_kwargs["dir"] = f_mkdir(callback_kwargs["output_dir"])
        wandb.init(**wandb_init_kwargs)
        self.num_processes = config.cfg.num_processes

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        **kwargs,
    ) -> None:
        """Called once train is supposed to log."""
        wandb.log({**self.wrap_metrics(metric_means)}, step=step)
        wandb.log(self.wrap_num_hard_resets(metrics), step=step)
        self.log_meta_from_task_data(tasks_data)

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Called after validation ends."""
        # has to log before logging videos
        # wandb.log({**self.wrap_metrics(metric_means)}, step=step)
        wandb.log({**self.wrap_metrics(metric_means), "valid-misc/step": step})
        if (
            "render" in kwargs.keys()
            and kwargs["render"] is not None
            and len(kwargs["render"]) != 0
        ):
            self.log_videos_from_render(mode="valid", render=kwargs["render"])

        self.log_2d_traj(mode="valid", traj_vizs=kwargs.get("extra_vizs", []))
        self.log_all_eval_point_clouds(tasks_data)

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Called after test ends."""
        # wandb.log({**self.wrap_metrics(metric_means)}, step=step)
        wandb.log(
            {**self.wrap_metrics(metric_means), "test-misc/step": step}, step=step
        )
        if (
            "render" in kwargs.keys()
            and kwargs["render"] is not None
            and len(kwargs["render"]) != 0
        ):
            self.log_videos_from_render(mode="test", render=kwargs["render"])
        self.log_2d_traj(mode="valid", traj_vizs=kwargs.get("extra_vizs", []))
        self.log_all_eval_point_clouds(tasks_data)

    def log_2d_traj(self, mode: str, traj_vizs: List[Dict[str, List[Figure]]]):
        idx = 0
        for viz_dict in traj_vizs:
            for k, figs in viz_dict.items():
                viz_method = "thor_trajectory" if "thor" in k else "base_trajectory"
                for fig in figs:
                    wandb.log(
                        {f"{mode}-{viz_method}/{self.run_name}_{idx}": wandb.Image(fig)}
                    )
                    idx += 1

    def after_save_project_state(self, base_dir: str) -> None:
        """Called after saving the project state in base_dir."""

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        """Determines the data returned to the `tasks_data` parameter in the
        above *_log functions."""
        sensors = []
        callback_sensors = self.cfg.callback_kwargs.get("callback_sensors", {}) or {}
        for sensor, sensor_kwargs in callback_sensors.items():
            # TODO configurable for metaworld envs
            if sensor == "video_callback":
                # for gym/metaworld envs
                sensors.append(TrainingVideoCallbackSensor(uuid="video_callback"))
            elif sensor == "point_cloud_callback":
                sensors.append(
                    CloudPointCallbackSensor(
                        uuid="point_cloud_callback", mode=self.mode
                    )
                )
            elif sensor == "thor_training_video_callback":
                if self.mode == "train":
                    _sensor_kwargs = self.cfg.sampler_kwargs.get("record_kwargs", {})
                    _sensor_kwargs.update(sensor_kwargs)
                    sensors.append(
                        ThorTrainingVideoCallbackSensor(
                            uuid="video_callback", **_sensor_kwargs
                        )
                    )
            elif sensor == "thor_point_cloud_callback":
                if self.mode == "train":
                    sensors.append(
                        ThorPointCloudCallbackSensor(uuid=sensor, **sensor_kwargs)
                    )
                # TODO maybe add for logging all valid
            else:
                raise NotImplementedError(sensor)
        return sensors

    def wrap_num_hard_resets(self, metrics: List[Dict[str, Any]]):
        if self.mode != "train":
            return {}
        wrapped_metrics = {}
        for metric in metrics:
            for key, value in metric.items():
                if "num_resets/" in key or "num_hard_resets/" in key:
                    process_ind = int(key.split("/")[-1])
                    old_val = self.saved_max_num_hard_reset_metrics[process_ind]
                    assert old_val <= value
                    self.saved_max_num_hard_reset_metrics[process_ind] = value
        if len(self.saved_max_num_hard_reset_metrics) == self.cfg.num_processes:
            total_resets = sum(self.saved_max_num_hard_reset_metrics.values())
            wrapped_metrics["num_resets/total_num_hard_resets"] = total_resets
            wrapped_metrics["num_resets/mean_num_hard_resets"] = (
                total_resets / self.cfg.num_processes
            )
        return wrapped_metrics

    @staticmethod
    def wrap_metrics(metric_means: Dict[str, float]):
        wrapped_metrics = {}
        for key, item in metric_means.items():
            split = key.split("/")
            if split[0] == "train-onpolicy-losses":
                loss_name = split[-1] if len(split) == 2 else f"{split[1]}_{split[-1]}"
                wrapped_metrics[f"{split[0]}/{loss_name}"] = item
            else:
                wrapped_metrics[key] = item
        return wrapped_metrics

    def log_meta_from_task_data(self, tasks_data: List[Dict[str, Optional[str]]]):
        """for recording during training."""
        for data in tasks_data:
            if (
                "video_callback" in data.keys()
                and data["video_callback"][0] is not None
            ):
                video_file, video_frames = data["video_callback"]
                split = video_file.split("/")
                # video_name = split[-2]    # process idx
                video_name = split[-1].split("-")[-1].split(".")[0]  # process idx
                recording_kwargs = self.cfg.sampler_kwargs.get("record_kwargs", {})
                wandb.log(
                    {
                        f"training-videos-{self.run_name}/{video_name}.mp4": wandb.Video(
                            # saved file path or the (L, C, H, W) np array if not saving
                            video_frames if video_frames is not None else video_file,
                            fps=recording_kwargs.get("fps", 30),
                            format="mp4",
                        )
                    },
                )
            elif data.get("point_cloud_callback", None) is not None:
                self._log_point_clouds(*data["point_cloud_callback"])
            elif data.get("thor_point_cloud_callback", None) is not None:
                wandb.log(
                    {
                        f"point_cloud_{self.run_name}": wandb.Object3D(
                            data["thor_point_cloud_callback"]
                        )
                    }
                )

    def _log_point_clouds(self, points: list, box: dict):
        point_clouds = {"type": "lidar/beta", "points": np.array(points)}
        if box is not None and len(box) > 0:
            point_clouds["boxes"] = np.array(box)
        # seems wandb fix the bug
        # JSON serializable
        # point_clouds["dump_encoder"] = NumpyJSONEncoder
        wandb.log({f"point_cloud_{self.run_name}": wandb.Object3D(point_clouds)})

    def log_videos_from_render(self, mode: str, render: dict):
        for key, item in render.items():
            images = []
            for img_dict in item:
                for img in img_dict.items():
                    images.append(img[1])
            # in L, C, H, W
            images = np.array(images).transpose([0, 3, 1, 2])
            wandb.log(
                {
                    f"{mode}-video/{self.run_name}-{key}": wandb.Video(
                        images, fps=self.cfg.viz_fps, format="mp4"
                    )
                }
            )

    def log_all_eval_point_clouds(self, task_data: List[Any]):
        clouds = []
        box = None
        for data in task_data:
            if data.get("point_cloud_callback", None) is not None:
                points, box = data["point_cloud_callback"]
                clouds.extend(points)
        if len(clouds) > 0:
            self._log_point_clouds(clouds, box)
