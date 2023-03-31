import glob
import os
import platform
from math import ceil
from typing import Dict, Optional, List, Any, Sequence, Union, Callable
from typing import Literal

import ai2thor
import ai2thor.build
import gym
import torch
import torch.optim as optim
from packaging import version
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from allenact.algorithms.onpolicy_sync.losses import losses_map
from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel
from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import Sensor, SensorSuite, ExpertActionSensor
from allenact.base_abstractions.task import TaskSampler
from allenact.embodiedai.aux_losses.losses import update_with_auxiliary_losses
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import (
    TrainingPipeline,
    Builder,
    PipelineStage,
    LinearDecay,
    evenly_distribute_count_into_bins,
    TrainingSettings,
)
from allenact.utils.system import get_logger
from allenact.utils.viz_utils import VizSuite, AgentViewViz, TrajectoryViz
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from allenact_plugins.robothor_plugin.robothor_sensors import (
    GPSCompassSensorRoboThor,
    DepthSensorThor,
)
from allenact_plugins.robothor_plugin.robothor_viz import ThorViz
from allenact_plugins.utils import *
from .env_wrappers import EnvWrapperType
from ..stretch_manipulathor_plugin.strech_sensors import StretchPolarSensor
from ..utils.config_utils import json_str
from ..utils.file_utils import f_exists

if (
    ai2thor.__version__ not in ["0.0.1", None]
    and not ai2thor.__version__.startswith("0+")
    and version.parse(ai2thor.__version__) < version.parse("3.2.0")
):
    raise ImportError(
        "To run the AI2-THOR ObjectNav baseline experiments you must use"
        " ai2thor version 3.2.0 or higher."
    )

import ai2thor.platform


__all__ = ["ExperimentConfigBase", "ThorExperimentConfigBase"]


class ExperimentConfigBase(ExperimentConfig):
    """Shared experiment config class.

    cls variables need to be defined:
        cfg : exp config defined by `Config` with config file path
        input_uuid : input obs sensor id
        phase_input_uuid : phase sensor id if using phase sensor
        ENV :  (gym) env name
        VALID_ENV: different env for eval, None for same as train
        SENSORS :  exp sensors list
        MODEL : exp model
        ACTION_SPACE : env action space
        TASK_SAMPLER : exp TaskSampler
    """

    cfg: Config
    input_uuid: Optional[str] = None
    phase_input_uuid: Optional[str] = None

    ENV: str
    EVAL_ENV: Optional[str] = None
    TEST_ENV: Optional[str] = None
    SENSORS: Optional[List[Sensor]] = None
    MODEL: ActorCriticModel
    TASK_SAMPLER: TaskSampler
    EVAL_TASK_SAMPLER: Optional[TaskSampler] = None
    ACTION_SPACE: Optional[gym.Space] = None
    # for generate action space automatically
    ENV_WRAPPER: Optional[EnvWrapperType] = None
    # for THOR
    THOR_COMMIT_ID: Optional[str] = None
    THOR_VALID_COMMIT_ID: Optional[str] = None
    THOR_TEST_COMMIT_ID: Optional[str] = None

    @classmethod
    def tag(cls) -> str:
        return cls.cfg.tag

    @classmethod
    def action_space(cls, env_wrapper: Optional[Any]):
        """directly get by dummy env if cls.ACTION_SPACE not specified but
        ENV_WRAPPER and ENV (name) is specified."""
        env_wrapper = env_wrapper or cls.ENV_WRAPPER
        dummy_env = env_wrapper(cls.ENV, is_dummy_env=True)
        action_space = dummy_env.action_space
        dummy_env.close()
        return action_space

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        # rgb uuid for Thor but not MuJuCo
        rgb_uuid = next((s.uuid for s in cls.SENSORS if isinstance(s, RGBSensor)), None)
        depth_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, DepthSensor)), None
        )
        goal_sensor_uuid = next(
            (
                s.uuid
                for s in cls.SENSORS
                if not isinstance(s, StretchPolarSensor)  # state obs sensor instead
                and isinstance(s, (GPSCompassSensorRoboThor, GoalObjectTypeThorSensor))
            ),
            None,
        )
        observation_space = kwargs.get(
            "sensor_preprocessor_graph", SensorSuite(cls.SENSORS)
        ).observation_spaces
        model_kwargs = dict(
            action_space=cls.ACTION_SPACE or cls.action_space(cls.ENV_WRAPPER),
            observation_space=observation_space,
        )
        if cls.input_uuid is not None:
            model_kwargs["input_uuid"] = cls.input_uuid
        if rgb_uuid is not None:
            model_kwargs["rgb_uuid"] = rgb_uuid
        if depth_uuid is not None:
            model_kwargs["depth_uuid"] = depth_uuid
        if goal_sensor_uuid is not None:
            model_kwargs["goal_uuid"] = goal_sensor_uuid
        model_kwargs.update(cls.cfg.model_kwargs)
        get_logger().debug(
            f"Initialize {cls.MODEL} with kwargs {json_str(model_kwargs)}"
        )
        return cls.MODEL(**model_kwargs)

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        if cls.EVAL_TASK_SAMPLER is not None and kwargs.get("mode", "train") != "train":
            # noinspection PyCallingNonCallable
            return cls.EVAL_TASK_SAMPLER(**kwargs)
        # noinspection PyCallingNonCallable
        return cls.TASK_SAMPLER(**kwargs)

    @staticmethod
    def if_none_next(*options) -> Optional[Any]:
        for opt in options:
            if opt is not None:
                return opt
        return None

    def _get_sampler_args(
        self,
        mode: Literal["train", "valid", "test"],
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate initialization arguments for train, valid, and test
        TaskSamplers for gym and/or MuJuCo envs.

        # Parameters
        process_ind : index of the current task sampler
        mode:  one of `train`, `valid`, or `test`
        seeds: seeds
        devices: device(s) for current sampler, in case specifying gpu rendering id in sampler
        """
        if mode == "train":
            env = self.ENV
            max_tasks = None  # infinite training tasks
            task_seeds_list = None  # no predefined random seeds for training
            deterministic_sampling = False  # randomly sample tasks in training
            sampler_kwargs = self.cfg.sampler_kwargs or {}
        elif mode == "valid":
            env = self.if_none_next(self.EVAL_ENV, self.cfg.eval_env, self.ENV)
            max_tasks = self.if_none_next(self.cfg.validation_tasks, 100)
            sampler_kwargs = self.if_none_next(
                self.cfg.valid_sampler_kwargs, self.cfg.sampler_kwargs, {}
            )
            # one seed for each task to sample:
            # - ensures different seeds for each sampler, and
            # - ensures a deterministic set of sampled tasks.
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )
            deterministic_sampling = True
        elif mode == "test":
            env = self.if_none_next(
                self.TEST_ENV,
                self.cfg.test_env,
                self.EVAL_ENV,
                self.cfg.eval_env,
                self.ENV,
            )
            # in case diff num of testing tasks (e.g. less valid for faster training process)
            max_tasks = self.if_none_next(
                self.cfg.testing_tasks, self.cfg.validation_tasks, 100
            )
            sampler_kwargs = self.if_none_next(
                self.cfg.test_sampler_kwargs,
                self.cfg.test_sampler_kwargs,
                self.cfg.sampler_kwargs,
                {},
            )
            task_seeds_list = list(
                range(process_ind * max_tasks, (process_ind + 1) * max_tasks)
            )
            deterministic_sampling = True
        else:
            raise NotImplementedError(mode)

        overwrite_sampler_kwargs = {**sampler_kwargs}
        if sampler_kwargs.get("record_during_training", False):
            num_record = (
                sampler_kwargs["num_record_processes"] or self.cfg.num_processes
            )
            if process_ind >= num_record:
                overwrite_sampler_kwargs["record_during_training"] = False

        return dict(
            gym_env_type=env,
            sensors=self.SENSORS,  # sensors used to return observations to the agent
            max_tasks=max_tasks,  # see above
            task_seeds_list=task_seeds_list,  # see above
            deterministic_sampling=deterministic_sampling,  # see above
            seed=seeds[process_ind],
            mode=mode,  # train, valid, or test
            devices=devices,
            process_ind=process_ind,
            **overwrite_sampler_kwargs,
            **kwargs,
        )

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            mode="train",
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            mode="valid",
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        return self._get_sampler_args(
            mode="test",
            process_ind=process_ind,
            total_processes=total_processes,
            devices=devices,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )

    def machine_params(
        self, mode: Literal["train", "valid", "test"] = "train", **kwargs
    ) -> Dict[str, Any]:
        sampler_devices: Sequence[int] = []
        use_gpu = self.cfg.num_gpu_use > 0
        if mode == "train":
            gpu_ids = [] if not use_gpu else self.cfg.train_gpus
            nprocesses = (
                self.cfg.num_processes
                if not use_gpu
                else evenly_distribute_count_into_bins(
                    self.cfg.num_processes, len(gpu_ids)
                )
            )
            with self.cfg.unlocked():
                self.cfg.num_gpu_use = len(self.cfg.train_gpus)  # in case modified
            sampler_devices = list(self.cfg.train_gpus)
        elif mode == "valid":
            # 0 for only checkpointing without eval
            gpu_ids = (
                self.cfg.validation_gpus
                if use_gpu and self.cfg.val_processes != 0
                else []
            )
            nprocesses = (
                self.cfg.val_processes
                if not use_gpu
                else evenly_distribute_count_into_bins(
                    self.cfg.val_processes, len(gpu_ids)
                )
            )
        elif mode == "test":
            gpu_ids = self.cfg.testing_gpus if use_gpu else []
            nprocesses = (
                self.cfg.test_processes
                if not use_gpu
                else evenly_distribute_count_into_bins(
                    self.cfg.test_processes, len(gpu_ids)
                )
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(self.SENSORS).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return {
            "nprocesses": nprocesses,
            "devices": gpu_ids,
            "sampler_devices": sampler_devices
            if mode == "train"
            else gpu_ids,  # ignored with > 1 gpu_ids
            "visualizer": self.get_viz(mode, self.cfg.viz_class),
            "sensor_preprocessor_graph": sensor_preprocessor_graph,
        }

    @classmethod
    def get_viz(
        cls,
        mode: Literal["train", "valid", "test"],
        viz_clss: Optional[Union[List[str], str]] = "video_viz",
    ):
        visualizer = None
        visualize = cls.cfg.visualize if mode == "valid" else cls.cfg.visualize_test
        if mode != "train" and visualize:
            vizs = {}
            viz_clss = viz_clss or "video_viz"
            if isinstance(viz_clss, str):
                viz_clss = [viz_clss]
            for viz in viz_clss:
                if viz == "video_viz":
                    render_kwargs = dict(mode=cls.cfg.viz_mode)
                    if cls.cfg.viz_resolution is not None:
                        render_kwargs["resolution"] = tuple(cls.cfg.viz_resolution)
                    vizs[viz] = AgentViewViz(
                        label="episode_vid",
                        max_clip_length=400,
                        vector_task_source=("render", render_kwargs,),
                    )
                elif viz == "base_trajectory":
                    get_logger().warning(f"maybe buggy, use callback instead")
                    vizs[viz] = TrajectoryViz(
                        path_to_target_location=None  # ("task_info", "path_to_target") # "target")
                    )
                elif viz == "thor_trajectory":
                    get_logger().warning(f"maybe buggy, use callback instead")
                    vizs[viz] = ThorViz(
                        # figsize=(16, 8),
                        # viz_rows_cols=(448, 448),
                        # scenes=(f"FloorPlan_{str_}"+"{}_{}", 1, 1, 1, 1),
                    )
                else:
                    raise NotImplementedError(viz)
            visualizer = VizSuite(mode=mode, **vizs)
        return visualizer

    @staticmethod
    def preprocessors():
        return tuple()

    @classmethod
    def training_pipeline(
        cls, loss_name: Optional[str] = None, **kwargs
    ) -> TrainingPipeline:
        loss_name = loss_name or cls.cfg.loss_name
        loss = losses_map[loss_name]
        named_losses = {loss_name: (loss(**cls.cfg.loss_kwargs), 1.0)}
        named_losses = update_with_auxiliary_losses(
            named_losses=named_losses,
            auxiliary_uuids=kwargs.get("auxiliary_uuids", []),
            multiple_beliefs=kwargs.get("multiple_beliefs", False),
        )

        loss_steps = cls.cfg.loss_steps
        if not isinstance(loss_steps, (int, float)):
            num_steps = list(cls.cfg.num_steps)
            loss_steps = list(loss_steps)
            assert len(loss_steps) == len(num_steps)
        else:
            loss_steps = [loss_steps]
            num_steps = [cls.cfg.num_steps]
        total_loss_steps = sum(loss_steps)

        pipeline_stages = [
            PipelineStage(
                loss_names=list(named_losses.keys()),
                max_stage_steps=loss_steps[i],
                training_settings=TrainingSettings(
                    **{**cls.cfg.training_setting_kwargs, **{"num_steps": num_steps[i]}}
                ),
            )
            for i in range(len(loss_steps))
        ]
        lr_schedule_steps = (
            cls.cfg.lr_scheduler_steps
            if cls.cfg.lr_scheduler_steps is not None
            else total_loss_steps
        )
        if cls.cfg.lr_scheduler == "linear":
            lr_scheduler_builder = Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=lr_schedule_steps,
                        startp=1.0,
                        endp=cls.cfg.end_lr / cls.cfg.lr,
                    )
                },
            )
        elif cls.cfg.lr_scheduler == "cos":
            lr_scheduler_builder = Builder(
                CosineAnnealingLR,
                {"T_max": lr_schedule_steps, "eta_min": cls.cfg.end_lr},
            )
        else:
            raise NotImplementedError(cls.cfg.lr_scheduler)

        return TrainingPipeline(
            training_settings=TrainingSettings(**cls.cfg.training_setting_kwargs),
            optimizer_builder=Builder(optim.Adam, dict(lr=cls.cfg.lr)),
            named_losses={key: val[0] for key, val in named_losses.items()},
            pipeline_stages=pipeline_stages,
            lr_scheduler_builder=lr_scheduler_builder,
        )


class ThorExperimentConfigBase(ExperimentConfigBase):
    """An Object Navigation experiment configuration in iThor."""

    TARGET_TYPES: Optional[Sequence[str]] = None
    EVAL_TARGET_TYPES: Optional[Sequence[str]] = None
    ENV_ARGS: Optional[dict] = None
    REWARD_CONFIG: Optional[dict] = None

    MAX_STEPS = 200  # 500 for objnav
    ENV_CLASS: Optional[Callable]
    TASK_CLASS: Optional[Callable]

    def __init__(
        self, env_args: Optional[dict] = None, reward_config: Optional[dict] = None
    ):
        self.ENV_ARGS = env_args
        self.REWARD_CONFIG = reward_config

    def get_dataset_dir(self, path: Optional[str] = None) -> Optional[str]:
        if path is None:
            get_logger().debug(
                f"dataset directory is not provided, may use "
                f"train dataset: {self.cfg.train_dataset_dir}, "
                f"validation dataset: {self.cfg.validation_dataset_dir},"
                f"or sample in the real-time"
            )
            return None
        if os.path.exists(path):
            return path
        return f_join(os.getcwd(), path)

    @property
    def scenes_sub_dir(self):
        return ""  # "episodes" for objnav

    @property
    def data_suffix(self):
        return ".json"  # ".json.gz"  for objnav

    @property
    def train_dataset_dir(self) -> str:
        return self.get_dataset_dir(self.cfg.train_dataset_dir)

    @property
    def val_dataset_dir(self) -> str:
        return (
            self.get_dataset_dir(self.cfg.validation_dataset_dir)
            or self.train_dataset_dir
        )

    @property
    def test_dataset_dir(self) -> str:
        return self.get_dataset_dir(self.cfg.test_dataset_dir) or self.val_dataset_dir

    def _get_sampler_args(
        self,
        mode: Literal["train", "valid", "test"],
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        if mode == "train":
            dataset_dir = self.train_dataset_dir
            num_task_per_scene = None
            allow_oversample = True
            cfg_sampler_kwargs = self.if_none_next(self.cfg.sampler_kwargs, {})
            thor_commit_id = self.THOR_COMMIT_ID
            task_seeds_list = None
            deterministic_sampling = False
        elif mode == "valid":
            dataset_dir = self.val_dataset_dir
            num_task_per_scene = self.cfg.validation_tasks
            allow_oversample = False
            cfg_sampler_kwargs = self.if_none_next(
                self.cfg.valid_sampler_kwargs, self.cfg.sampler_kwargs, {}
            )
            thor_commit_id = self.if_none_next(
                self.THOR_VALID_COMMIT_ID, self.THOR_COMMIT_ID
            )
            if num_task_per_scene is not None:
                task_seeds_list = list(
                    range(
                        process_ind * num_task_per_scene,
                        (process_ind + 1) * num_task_per_scene,
                    )
                )
            else:
                task_seeds_list = None
            deterministic_sampling = True
        elif mode == "test":
            dataset_dir = self.test_dataset_dir
            num_task_per_scene = self.cfg.testing_tasks
            allow_oversample = False
            cfg_sampler_kwargs = self.if_none_next(
                self.cfg.test_sampler_kwargs,
                self.cfg.valid_sampler_kwargs,
                self.cfg.sampler_kwargs,
                {},
            )
            thor_commit_id = self.if_none_next(
                self.THOR_TEST_COMMIT_ID, self.THOR_VALID_COMMIT_ID, self.THOR_COMMIT_ID
            )
            if num_task_per_scene is not None:
                task_seeds_list = list(
                    range(
                        process_ind * num_task_per_scene,
                        (process_ind + 1) * num_task_per_scene,
                    )
                )
            else:
                task_seeds_list = None
            deterministic_sampling = True
        else:
            raise ValueError(mode)
        if dataset_dir is not None:
            scenes_dir = f_join(dataset_dir, self.scenes_sub_dir)
            path = f_join(scenes_dir, f"*{self.data_suffix}")
            scenes = [scene.split("/")[-1].split(".")[0] for scene in glob.glob(path)]
            if len(scenes) == 0:
                raise RuntimeError(
                    (
                        "Could find no scene dataset information in directory {}."
                        " Are you sure you've downloaded them? "
                        " If not, see https://allenact.org/installation/download-datasets/ information"
                        " on how this can be done."
                    ).format(scenes_dir)
                )

            oversample_warning = (
                f"Warning: oversampling some of the scenes ({scenes}) to feed all processes ({total_processes})."
                " You can avoid this by setting a number of workers divisible by the number of scenes"
            )
            if total_processes > len(scenes):  # oversample some scenes -> bias
                if not allow_oversample:
                    raise RuntimeError(
                        f"Cannot have `total_processes > len(scenes)`"
                        f" ({total_processes} > {len(scenes)}) when `allow_oversample` is `False`."
                    )
                if total_processes % len(scenes) != 0:
                    get_logger().warning(oversample_warning)
                scenes = scenes * int(ceil(total_processes / len(scenes)))
                scenes = scenes[: total_processes * (len(scenes) // total_processes)]
            elif len(scenes) % total_processes != 0:
                get_logger().warning(oversample_warning)

            inds = partition_inds(len(scenes), total_processes)
        else:
            # real-time sampling w/o dataset
            scenes, inds = None, None

        # make each process relative fixed the object types for reset free learning
        if mode == "train":
            object_types = self.TARGET_TYPES
        else:
            object_types = self.if_none_next(self.EVAL_TARGET_TYPES, self.TARGET_TYPES)
        if (
            isinstance(object_types, tuple)
            and total_processes > 1
            and not total_processes % len(object_types) not in [0, len(object_types)]
        ):
            get_logger().debug(
                f"more than one ({object_types}) object type will be sampled for each process "
                f"{total_processes}: {object_types} probably not expected in reset free learning"
            )
        if isinstance(object_types[0], tuple):
            # stretch robot
            object_types = object_types * int(ceil(total_processes / len(object_types)))
            obj_inds = partition_inds(len(object_types), total_processes)
            object_types = object_types[
                obj_inds[process_ind] : obj_inds[process_ind + 1]
            ]

        if not self.cfg.headless:
            x_display: Optional[str] = None
            if platform.system() == "Linux":
                x_displays = get_open_x_displays(throw_error_if_empty=True)
                n_devices = len([d for d in devices if d != torch.device("cpu")])
                if n_devices > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                elif n_devices < len(x_displays) and len(devices) == 1:
                    # TODO check
                    x_display = x_displays[devices[0]]
                else:
                    x_display = x_displays[process_ind % len(x_displays)]
            device_dict = dict(x_display=x_display)
        else:
            device_dict = dict(
                gpu_device=devices[process_ind % len(devices)],
                platform=ai2thor.platform.CloudRendering,
            )

        overwrite_sampler_kwargs = {**cfg_sampler_kwargs}
        if cfg_sampler_kwargs.get("record_during_training", False):
            num_record = (
                cfg_sampler_kwargs["num_record_processes"] or self.cfg.num_processes
            )
            if process_ind >= num_record:
                overwrite_sampler_kwargs["record_during_training"] = False

        sampler_kwargs: Dict[str, Union[Any]] = {
            # some of them are for robothor objnav
            "mode": mode,
            "scene_directory": dataset_dir,
            "loop_dataset": mode == "train",
            "allow_flipping": mode == "train",
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]]
            if scenes is not None
            else scenes,
            "object_types": object_types,
            "max_steps": self.MAX_STEPS,
            "sensors": [
                s
                for s in self.SENSORS
                if (mode == "train" or not isinstance(s, ExpertActionSensor))
            ],
            "action_space": self.ACTION_SPACE or self.action_space(self.ENV_WRAPPER),
            "seed": seeds[process_ind] if seeds is not None else None,
            "task_seeds_list": task_seeds_list,
            "deterministic_sampling": deterministic_sampling,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": self.REWARD_CONFIG,
            "env_args": {
                **self.ENV_ARGS,
                **device_dict,
                "renderDepthImage": any(
                    isinstance(s, DepthSensorThor) for s in self.SENSORS
                ),
            },
            "num_task_per_scene": num_task_per_scene,
            "max_tasks": num_task_per_scene,
            "process_ind": process_ind,
            **kwargs,
        }
        if self.ENV_CLASS is not None:
            sampler_kwargs["env_class"] = self.ENV_CLASS
        if self.TASK_CLASS is not None:
            sampler_kwargs["task_class"] = self.TASK_CLASS

        if thor_commit_id is not None:
            if ".app" in thor_commit_id and "AI2-THOR" in thor_commit_id:
                # use local build instead
                assert f_exists(thor_commit_id), thor_commit_id
                sampler_kwargs["env_args"].pop("commit_id")
                sampler_kwargs["env_args"]["local_executable_path"] = thor_commit_id
            else:
                sampler_kwargs["env_args"]["commit_id"] = (
                    thor_commit_id if not self.cfg.headless else ai2thor.build.COMMIT_ID
                )

        sampler_kwargs.update(overwrite_sampler_kwargs)
        return sampler_kwargs

    # def test_task_sampler_args(
    #     self,
    #     process_ind: int,
    #     total_processes: int,
    #     devices: Optional[List[int]] = None,
    #     seeds: Optional[List[int]] = None,
    #     deterministic_cudnn: bool = False,
    # ) -> Dict[str, Any]:
    # if (
    #     self.test_dataset_dir == self.val_dataset_dir
    #     and self.val_dataset_dir is not None
    # ):
    #     get_logger().warning(
    #         "No test dataset dir detected, running test on validation set instead."
    #         " Be careful as the saved metrics json and tensorboard files *will still be labeled as"
    #         " 'test' rather than 'valid'**."
    #     )
    #     return self.valid_task_sampler_args(
    #         process_ind=process_ind,
    #         total_processes=total_processes,
    #         devices=devices,
    #         seeds=seeds,
    #         deterministic_cudnn=deterministic_cudnn,
    #     )
    #
    # else:
    #     res = self.valid_task_sampler_args(
    #         process_ind=process_ind,
    #         total_processes=total_processes,
    #         devices=devices,
    #         seeds=seeds,
    #         deterministic_cudnn=deterministic_cudnn,
    #     )
    #     res["env_args"]["all_metadata_available"] = False
    #     res["rewards_config"] = {**res["rewards_config"], "shaping_weight": 0}
    #     return res
