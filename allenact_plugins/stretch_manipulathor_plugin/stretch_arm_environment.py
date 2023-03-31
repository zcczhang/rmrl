import copy
import math
import random
import typing
from typing import Any, Optional, Dict, Union, cast, Callable, Sequence, Tuple
from typing import OrderedDict as OrderedDictType

import numpy as np
from ai2thor.controller import Controller
from ai2thor.server import Event
from torch.distributions.utils import lazy_property

from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import *
from allenact_plugins.stretch_manipulathor_plugin.stretch_sim2real_utils import (
    kinect_reshape,
    intel_reshape,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_utils import (
    remove_nan_inf_for_frames,
    get_relative_stretch_current_arm_state,
    make_all_objects_unbreakable,
    get_abs_obj1_from_rel_obj1_to_obj2,
    sample_obj_pos_over_space,
)

__all__ = ["StretchManipulaTHOREnvironment"]


class StretchManipulaTHOREnvironment(IThorEnvironment):

    # noinspection PyMissingConstructor
    def __init__(
        self,
        # task-specific
        randomize_materials: bool = False,
        randomize_lighting: bool = False,
        controllable_randomization: bool = True,
        init_teleport_kwargs: Optional[dict] = None,
        init_teleport_random_bounds: Optional[tuple] = (-0.2, 0.2),
        hand_sphere_radius: Optional[float] = None,
        docker_enabled: bool = False,
        local_thor_build: Optional[str] = None,
        visibility_distance: float = STRETCH_VISIBILITY_DISTANCE,
        quality: str = "Very Low",
        restrict_to_initially_reachable_points: bool = False,
        object_open_speed: float = 0.05,
        simplify_physics: bool = False,
        verbose: bool = False,
        env_args: Optional[dict] = None,
    ) -> None:
        self.reset_extra_commands: OrderedDictType[
            str, Union[dict, Callable]
        ] = OrderedDict(
            MakeAllObjectsMoveable=dict(action="MakeAllObjectsMoveable"),
            MakeObjectsStaticKinematicMassThreshold=dict(
                action="MakeObjectsStaticKinematicMassThreshold"
            ),
            make_all_objects_unbreakable=make_all_objects_unbreakable,
            MoveArm=dict(
                action="MoveArm", position=dict(x=0, y=0.8, z=0), **ADDITIONAL_ARM_ARGS
            ),
            # latest commit robot starting from hand facing itself for FloorPlan{i}
            RotateWrist=dict(action="RotateWristRelative", yaw=180),
            AddThirdPartyCamera={},  # placeholder for third-party cameras
        )
        self.randomize_materials = randomize_materials
        self.randomize_lighting = randomize_lighting
        self.controllable_randomization = controllable_randomization
        # for deterministic visual randomization
        self.current_visual_randomization = None
        if not controllable_randomization and randomize_materials:
            self.reset_extra_commands["RandomizeMaterials"] = dict(
                action="RandomizeMaterials"
            )
        if not controllable_randomization and randomize_lighting:
            self.reset_extra_commands["RandomizeLighting"] = dict(
                action="RandomizeLighting"
            )
        self.init_teleport_kwargs = init_teleport_kwargs
        self.init_teleport_random_bounds = init_teleport_random_bounds or (0, 0)

        if hand_sphere_radius is not None:
            assert (
                isinstance(hand_sphere_radius, float)
                and 0.04 <= hand_sphere_radius <= 0.5
            )
            self.reset_extra_commands["SetHandSphereRadius"] = dict(
                action="SetHandSphereRadius", radius=hand_sphere_radius
            )

        self._start_player_screen_width = env_args["width"]
        self._start_player_screen_height = env_args["height"]
        self._local_thor_build = local_thor_build
        # self.x_display = x_display
        # self.controller: Optional[Controller] = None
        self._started = False
        self._quality = quality
        self._verbose = verbose
        self.env_args = env_args

        self._initially_reachable_points: Optional[typing.List[Dict]] = None
        self._initially_reachable_points_set: Optional[
            typing.Set[typing.Tuple[float, float]]
        ] = None
        self._move_mag: Optional[float] = None
        self._grid_size: Optional[float] = None
        self._visibility_distance = visibility_distance

        self.restrict_to_initially_reachable_points = (
            restrict_to_initially_reachable_points
        )
        self.object_open_speed = object_open_speed
        self.simplify_physics = simplify_physics

        self.docker_enabled = docker_enabled

        if "quality" not in self.env_args:
            self.env_args["quality"] = self._quality

        self.added_third_party_camera_info = None

        # this should be increased in task instance,
        # in case step with other actions specifically for controllers
        self.cum_env_steps = 0  # cumulative steps once initialized
        self.cum_steps_for_reset = 0  # cumulative steps each reset
        self.num_resets = 0
        self.pick_release_correctly = None

    def add_third_party_camera(
        self,
        name: str,
        position: dict,
        rotation: dict,
        skyboxColor: str = "white",
        update_if_existed: bool = False,
        add_every_reset: bool = False,
        **kwargs,
    ):
        """track and organize cameras."""
        self.added_third_party_camera_info = self.added_third_party_camera_info or {}
        if name in self.added_third_party_camera_info:
            if not update_if_existed:
                get_logger().warning(f"{name} camera already in use, not updated")
                return
            get_logger().warning(f"{name} camera already in use, updated")
            action_dict = dict(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=self.added_third_party_camera_info[name],
            )
        else:
            action_dict = dict(action="AddThirdPartyCamera")
            # as idx start from 0, new camera append to last
            camera_id = len(self.last_event.third_party_camera_frames)
            self.added_third_party_camera_info[name] = camera_id

        camera_kwargs = dict(
            position=position, rotation=rotation, skyboxColor=skyboxColor, **kwargs
        )
        action_dict.update(camera_kwargs)
        self.controller.step(**action_dict)

        if add_every_reset:
            self.reset_extra_commands["AddThirdPartyCamera"][name] = dict(
                action="AddThirdPartyCamera", **camera_kwargs
            )

    @lazy_property
    def controller(self) -> Controller:
        self._started = True
        controller = self.create_controller()
        self.check_controller_version(controller)
        controller.docker_enabled = self.docker_enabled
        return controller

    def create_controller(self) -> Controller:
        assert (
            "commit_id" in self.env_args or "local_executable_path" in self.env_args
        ), self.env_args
        controller = Controller(**self.env_args)
        return controller

    def start(
        self, scene_name: Optional[str], move_mag: float = 0.25, **kwargs,
    ) -> None:
        raise Exception("Should not be called")

    def increment_steps_stats(self):
        """call in task instance."""
        self.cum_env_steps += 1
        self.cum_steps_for_reset += 1

    def update_reset_stats(self):
        """call in task_sampler."""
        self.num_resets += 1
        self.cum_steps_for_reset = 0

    def randomize_texture_lightning(
        self,
        *,
        table_color: Optional[Sequence[int]] = None,
        table_leg_color: Optional[Sequence[int]] = None,
        table_leg_material: Optional[int] = None,
        wall_color: Optional[Sequence[int]] = None,
        wall_material: Optional[int] = None,
        floor_color: Optional[Sequence[int]] = None,
        floor_material: Optional[int] = None,
        light_color: Optional[Sequence[int]] = None,
        light_intensity: Optional[Union[float, int]] = None,
        texture_randomization_keys: Union[str, Tuple[str]] = ("table", "wall", "floor"),
    ):
        assert self.controllable_randomization
        if isinstance(texture_randomization_keys, str):
            texture_randomization_keys = (texture_randomization_keys,)
        assert set(texture_randomization_keys).issubset(
            {"table", "wall", "floor"}
        ), texture_randomization_keys

        def _sample_color(color: Optional[Sequence[int]]) -> Sequence[int]:
            if color is None:
                return [random.randint(0, 255) for _ in range(3)]
            else:
                # assert all(0 <= i <= 255 for i in color)
                return color

        def _sample_material(material: Optional[int]) -> int:
            if material is None:
                return random.randint(0, 4)
            else:
                assert 0 <= material <= 4
                return int(material)

        if self.randomize_materials:
            if "table" in texture_randomization_keys:
                table_color = _sample_color(table_color)
                self.controller.step(
                    dict(
                        action="ChangeTableTopColorExpRoom",
                        r=table_color[0],
                        g=table_color[1],
                        b=table_color[2],
                    )
                )
                table_leg_material = _sample_material(table_leg_material)
                self.controller.step(
                    dict(
                        action="ChangeTableLegMaterialExpRoom",
                        objectVariation=table_leg_material,
                    )
                )
                table_leg_color = _sample_color(table_leg_color)
                self.controller.step(
                    dict(
                        action="ChangeTableLegColorExpRoom",
                        r=table_leg_color[0],
                        g=table_leg_color[1],
                        b=table_leg_color[2],
                    )
                )
            if "wall" in texture_randomization_keys:
                wall_material = _sample_material(wall_material)
                self.controller.step(
                    dict(
                        action="ChangeWallMaterialExpRoom",
                        objectVariation=wall_material,
                    )
                )
                wall_color = _sample_color(wall_color)
                self.controller.step(
                    dict(
                        action="ChangeWallColorExpRoom",
                        r=wall_color[0],
                        g=wall_color[1],
                        b=wall_color[2],
                    )
                )
            if "floor" in texture_randomization_keys:
                floor_material = _sample_material(floor_material)
                self.controller.step(
                    dict(
                        action="ChangeFloorMaterialExpRoom",
                        objectVariation=floor_material,
                    )
                )
                floor_color = _sample_color(floor_color)
                self.controller.step(
                    dict(
                        action="ChangeFloorColorExpRoom",
                        r=floor_color[0],
                        g=floor_color[1],
                        b=floor_color[2],
                    )
                )

        if self.randomize_lighting:
            if light_intensity is None:
                light_intensity = random.uniform(0.5, 2)
            # light color also has chance to make the env darker
            assert 0.5 <= light_intensity <= 2
            self.controller.step(
                dict(action="ChangeLightIntensityExpRoom", intensity=light_intensity)
            )
            light_color = _sample_color(light_color)
            self.controller.step(
                dict(
                    action="ChangeLightColorExpRoom",
                    r=light_color[0],
                    g=light_color[1],
                    b=light_color[2],
                )
            )

        # update current randomization
        self.current_visual_randomization = dict(
            table_color=table_color,
            table_leg_color=table_leg_color,
            table_leg_material=table_leg_material,
            wall_color=wall_color,
            wall_material=wall_material,
            floor_color=floor_color,
            floor_material=floor_material,
            light_color=light_color,
            light_intensity=light_intensity,
        )

    def reset(
        self,
        scene_name: Optional[str] = None,
        move_mag: float = 0.25,
        *,
        visual_randomize_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """reset to initial state, NO sampling and teleporting."""
        self._move_mag = move_mag
        self._grid_size = self._move_mag

        if scene_name is None:
            scene_name = self.controller.last_event.metadata["sceneName"]

        try:
            self.reset_environment_and_additional_commands(
                scene_name, visual_randomize_kwargs
            )
        except Exception as e:
            get_logger().warning(
                "RESETTING THE SCENE,", scene_name, "because of", str(e)
            )
            self.controller = self.create_controller()
            self.reset_environment_and_additional_commands(
                scene_name, visual_randomize_kwargs
            )
        # below is not necessary in ExpRoom though
        if self.object_open_speed != 1.0:
            self.controller.step(
                {"action": "ChangeOpenSpeed", "x": self.object_open_speed}
            )

        self._initially_reachable_points = None
        self._initially_reachable_points_set = None
        self.controller.step({"action": "GetReachablePositions"})
        if (
            not self.controller.last_event.metadata["lastActionSuccess"]
            and self.scene_name != "FloorPlan_ExpRoom"
        ):
            get_logger().warning(
                "Error when getting reachable points: {}".format(
                    self.controller.last_event.metadata["errorMessage"]
                )
            )
        self._initially_reachable_points = self.last_action_return
        # call in task sampler
        # self.update_reset_stats()

    def reset_environment_and_additional_commands(
        self,
        scene_name: Optional[str] = None,
        visual_randomize_kwargs: Optional[dict] = None,
    ):
        if scene_name == "Procedural":
            return
        else:
            self.controller.reset(scene_name)
            new_camera_info = {}
            cam_frame_len = len(self.last_event.third_party_camera_frames)
            if cam_frame_len >= 1:
                new_camera_info["kinect"] = 0
                if cam_frame_len == 2:
                    new_camera_info["wrist_centric"] = 1
            for command_name, extra_command in self.reset_extra_commands.items():
                if command_name == "make_all_objects_unbreakable":
                    extra_command(self.controller)
                elif command_name == "AddThirdPartyCamera":
                    for cam_name, cam_kwargs in extra_command.items():
                        self.controller.step(cam_kwargs)
                        # put index to the dict for fixed using cameras
                        new_camera_info[cam_name] = self.added_third_party_camera_info[
                            cam_name
                        ]
                else:
                    event = self.controller.step(extra_command)
                    if event.metadata["lastActionSuccess"] is False:
                        get_logger().warning(
                            f"Failed {extra_command} as {event.metadata['errorMessage']}"
                        )
            assert self.controller.last_event.metadata["fov"] == max(
                INTEL_FOV_W, INTEL_FOV_H
            )
            assert self.controller.last_event.metadata["thirdPartyCameras"][0][
                "fieldOfView"
            ] == max(KINECT_FOV_W, KINECT_FOV_H)

            if (
                self.init_teleport_kwargs is not None
                and len(self.init_teleport_kwargs) > 0
            ):
                teleport_kwargs = copy.deepcopy(self.init_teleport_kwargs)
                # in FloorPlan_ExpRoom
                assert self.scene_name == "FloorPlan_ExpRoom"
                teleport_kwargs["x"] += random.uniform(
                    *self.init_teleport_random_bounds
                )
                self.teleport_agent_to(**teleport_kwargs)
            self.current_visual_randomization = None
            if self.controllable_randomization and (
                self.randomize_lighting or self.randomize_materials
            ):
                self.randomize_texture_lightning(**visual_randomize_kwargs)
            self.added_third_party_camera_info = new_camera_info
            self.wait()

    def check_controller_version(self, controller: Optional[Controller] = None):
        controller_to_check = controller
        if controller_to_check is None:
            controller_to_check = self.controller
        if (
            "local_executable_path" not in self.env_args
            and STRETCH_MANIPULATHOR_COMMIT_ID is not None
        ):
            assert self.env_args["commit_id"] in controller_to_check._build.url, (
                "Build number is not right, {} vs {}, use  "
                "pip3 install -e git+https://github.com/allenai/ai2thor.git@{}#egg=ai2thor".format(
                    controller_to_check._build.url,
                    STRETCH_MANIPULATHOR_COMMIT_ID,
                    STRETCH_MANIPULATHOR_COMMIT_ID,
                )
            )

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        raise NotImplementedError("not used")

    def random_sample_objs_over_space(
        self,
        obj_ids: list,
        space: dict,
        sampling_limit: int = 1000,
        raise_error: bool = False,
    ):
        objs_info = self.get_objects_by_ids(obj_ids)
        return sample_obj_pos_over_space(
            objs_info=objs_info,
            space=space,
            sampling_limit=sampling_limit,
            raise_error=raise_error,
        )

    def is_object_at_low_level_hand(self, object_id: str):
        current_objects_in_hand = self.last_event.metadata["arm"]["heldObjects"]
        return object_id in current_objects_in_hand

    def object_in_hand(self):
        """Object id(s) for the object in the agent's hand."""
        inv_objs = self.last_event.metadata["arm"]["heldObjects"]
        if len(inv_objs) == 0:
            return None
        elif len(inv_objs) == 1:
            return self.get_object_by_id(inv_objs[0])
        else:
            raise AttributeError("Must be <= 1 inventory objects.")

    def get_object_by_id(
        self, object_id: str, warning: bool = True
    ) -> Optional[Dict[str, Any]]:
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] == object_id:
                o["position"] = self.correct_nan_inf(o["position"], "obj id")
                return o
        debug_msg = (
            f"no {object_id} in "
            f"{[obj['objectId'] for obj in self.last_event.metadata['objects']]}"
        )
        if warning:
            get_logger().warning(debug_msg)
        return None

    def get_objects_by_ids(
        self, object_ids: Union[list, str], warning: bool = True
    ) -> Optional[Dict[str, dict]]:
        if isinstance(object_ids, str):
            object_ids = [object_ids]
        obj_infos = {}
        for o in self.last_event.metadata["objects"]:
            if o["objectId"] in object_ids:
                o["position"] = self.correct_nan_inf(o["position"], "obj id")
                obj_infos[o["objectId"]] = o
            if set(obj_infos.keys()) == set(object_ids):
                return obj_infos
        debug_msg = (
            f"did not find all {object_ids} in "
            f"{[obj['objectId'] for obj in self.last_event.metadata['objects']]}"
        )
        if warning:
            get_logger().warning(debug_msg)
        return None

    def get_object_by_type(self, object_type: str) -> list:
        return self.last_event.objects_by_type(object_type)

    def get_object_by_name(self, object_name: str) -> list:
        if "|" in object_name:
            get_logger().warning(
                f"input is object id: {object_name}, use `get_object_by_id` instead"
            )
            return [self.get_object_by_id(object_name)]
        return [
            obj
            for obj in self.last_event.metadata["objects"]
            if obj["name"] == object_name
        ]

    def get_reachable_positions(self) -> list:
        """agent reachable positions."""
        event = self.controller.step("GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]

        if reachable_positions is None or len(reachable_positions) == 0:
            get_logger().warning(
                "Scene name", self.controller.last_event.metadata["sceneName"]
            )
            if self.controller.last_event.metadata["sceneName"] == "Procedural":
                return []
            else:
                get_logger().warning(self.last_event.metadata["errorMessage"])
                # pdb.set_trace()
        return reachable_positions

    def get_current_arm_state(self):
        raise NotImplementedError("not used, use `get_absolute_hand_state` instead")

    def get_absolute_hand_state(self) -> dict:
        """hand center (not wrist)"""
        arm = self.controller.last_event.metadata["arm"]
        hand_center = arm["handSphereCenter"]
        return dict(
            position=self.correct_nan_inf(hand_center, "handSphereCenter"),
            rotation={"x": 0, "y": 0, "z": 0},
            radius=arm["handSphereRadius"],
        )

    def set_hand_sphere_radius(self, radius: float):
        assert 0.04 <= radius <= 0.5 and isinstance(radius, float)
        return self.controller.step(dict(action="SetHandSphereRadius", radius=radius))

    def get_pickupable_objects(self):
        """pickupable objects in the current state (view)."""
        event = self.controller.last_event
        object_list = event.metadata["arm"]["pickupableObjects"]
        return object_list

    def get_current_object_locations(self) -> dict:
        obj_loc_dict = {}
        metadata = self.controller.last_event.metadata["objects"]
        for o in metadata:
            obj_loc_dict[o["objectId"]] = dict(
                position=o["position"], rotation=o["rotation"]
            )
        return copy.deepcopy(obj_loc_dict)

    def teleport_object(
        self,
        target_object_id: str,
        target_position: dict,
        rotation: Optional[Union[float, int, dict]] = None,
        fixed: bool = False,
    ) -> Event:
        teleport_kwargs = dict(
            action="PlaceObjectAtPoint",
            objectId=target_object_id,
            position=target_position,
            # do in below to make it stationary
            # forceKinematic=fixed
        )
        if rotation is not None:
            if isinstance(rotation, (int, float)):
                # obj in table is supposed to be only having y rotation
                rotation = dict(x=0, y=rotation, z=0)
            teleport_kwargs["rotation"] = rotation
        event = self.controller.step(teleport_kwargs)
        if not event.metadata["lastActionSuccess"]:
            get_logger().warning(
                f"failed teleport {target_object_id} "
                f"from {self.get_object_by_id(target_object_id)['position']} to {target_position}, "
                f"with error message: {event.metadata['errorMessage']}"
            )
        if fixed:
            stationary_event = self.controller.step(
                dict(action="MakeSpecificObjectStationary", objectId=target_object_id)
            )
            # return teleport even only unless making obj stationary failed
            if not stationary_event.metadata["lastActionSuccess"]:
                event = stationary_event
        self.wait()
        return event

    def get_all_receptacles(self, id_only: bool = False) -> list:
        return [
            (obj["objectId"] if id_only else obj)
            for obj in self.all_objects()
            if obj["receptacle"] and len(obj["receptacleObjectIds"]) > 0
        ]

    def get_parent_receptacles(self, target_obj: str) -> set:
        assert "|" in target_obj, f"input {target_obj} should be object id"
        all_containing_receptacle = set([])
        parent_queue = [target_obj]
        while len(parent_queue) > 0:
            top_queue = parent_queue[0]
            parent_queue = parent_queue[1:]
            if top_queue in all_containing_receptacle:
                continue
            current_parent_list = self.last_event.get_object(top_queue)[
                "parentReceptacles"
            ]
            if current_parent_list is None:
                continue
            else:
                parent_queue += current_parent_list
                all_containing_receptacle.update(set(current_parent_list))
        return all_containing_receptacle

    def is_object_in_receptacle(self, target_obj: str, target_receptacle: str) -> bool:
        return target_receptacle in self.get_parent_receptacles(target_obj)

    def step(
        self,
        action_dict: Optional[Dict[str, Union[str, int, float, Dict]]] = None,
        **kwargs: Union[str, int, float, Dict],
    ) -> Event:
        """Take a step in the ai2thor environment."""
        action = cast(str, action_dict["action"])

        skip_render = "renderImage" in action_dict and not action_dict["renderImage"]
        last_frame: Optional[np.ndarray] = None
        if skip_render:
            last_frame = self.current_frame

        action_success = None
        self.pick_release_correctly = None

        if self.simplify_physics:
            action_dict["simplifyPhysics"] = True
        if action in [PICKUP, DONE, RELEASE]:
            self.pick_release_correctly = False
            if action == PICKUP:
                object_id = action_dict["object_id"]
                if not self.is_object_at_low_level_hand(object_id):
                    pickupable_objects = self.get_pickupable_objects()
                    if object_id in pickupable_objects:
                        self.controller.step(
                            dict(action="PickupObject", objectIdCandidates=[object_id])
                        )
                        # just for debugging, could del when confident
                        cur_hand_inv = self.last_event.metadata["arm"]["heldObjects"]
                        action_success = self.controller.last_event.metadata[
                            "lastActionSuccess"
                        ]
                        if not action_success or [object_id] != cur_hand_inv:
                            get_logger().warning(
                                f'unexpected pickup {object_id}: {self.last_event.metadata["arm"]["heldObjects"]}, '
                                f'errorMessage: {self.last_event.metadata["errorMessage"]},'
                            )
                        else:
                            self.pick_release_correctly = True
            elif action == RELEASE:
                self.controller.step(dict(action="ReleaseObject"))
                action_success = self.controller.last_event.metadata[
                    "lastActionSuccess"
                ]
                if not action_success:
                    get_logger().warning(
                        f"{RELEASE} failed as {self.controller.last_event.metadata['errorMessage']}"
                    )
                else:
                    self.pick_release_correctly = True
            # action_dict = {"action": "Pass"}
            action_dict = dict(action="AdvancePhysicsStep", simSeconds=1.0)
        elif action in [MOVE_AHEAD, MOVE_BACK, ROTATE_LEFT, ROTATE_RIGHT]:
            copy_aditions = copy.deepcopy(ADDITIONAL_ARM_ARGS)
            action_dict = {**action_dict, **copy_aditions}
            if action == MOVE_AHEAD:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = AGENT_MOVEMENT_CONSTANT
            elif action == MOVE_BACK:
                action_dict["action"] = "MoveAgent"
                action_dict["ahead"] = -AGENT_MOVEMENT_CONSTANT
            elif action == ROTATE_RIGHT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = AGENT_ROTATION_DEG

            elif action == ROTATE_LEFT:
                action_dict["action"] = "RotateAgent"
                action_dict["degrees"] = -AGENT_ROTATION_DEG
        elif action in [
            MOVE_ARM_HEIGHT_P,
            MOVE_ARM_HEIGHT_M,
            MOVE_ARM_Z_P,
            MOVE_ARM_Z_M,
        ]:
            base_position = get_relative_stretch_current_arm_state(self.controller)
            change_value = ARM_MOVE_CONSTANT
            if action == MOVE_ARM_HEIGHT_P:
                base_position["y"] += change_value
            elif action == MOVE_ARM_HEIGHT_M:
                base_position["y"] -= change_value
            elif action == MOVE_ARM_Z_P:
                base_position["z"] += change_value
            elif action == MOVE_ARM_Z_M:
                base_position["z"] -= change_value
            action_dict = dict(
                action="MoveArm",
                position=dict(
                    x=base_position["x"], y=base_position["y"], z=base_position["z"]
                ),
                **ADDITIONAL_ARM_ARGS,
            )
        elif action in [MOVE_WRIST_P, MOVE_WRIST_M]:
            if action == MOVE_WRIST_P:
                action_dict = dict(action="RotateWristRelative", yaw=-WRIST_ROTATION)
            elif action == MOVE_WRIST_M:
                action_dict = dict(action="RotateWristRelative", yaw=WRIST_ROTATION)
        elif action == MOVE_WRIST_P_SMALL:
            action_dict = dict(action="RotateWristRelative", yaw=-WRIST_ROTATION / 5)
        elif action == MOVE_WRIST_M_SMALL:
            action_dict = dict(action="RotateWristRelative", yaw=WRIST_ROTATION / 5)
        elif action == ROTATE_LEFT_SMALL:
            action_dict["action"] = "RotateAgent"
            action_dict["degrees"] = -AGENT_ROTATION_DEG / 5
        elif action == ROTATE_RIGHT_SMALL:
            action_dict["action"] = "RotateAgent"
            action_dict["degrees"] = AGENT_ROTATION_DEG / 5

        sr = self.controller.step(action_dict)

        if action_success is not None:
            self.last_action_success = action_success

        if self.restrict_to_initially_reachable_points:
            self._snap_agent_to_initially_reachable()

        if skip_render:
            assert last_frame is not None
            self.last_event.frame = last_frame

        return sr

    def hand_reachable_space_on_countertop(
        self, countertop_id: str, fixed_axis: str = "z", offset: float = 0.1
    ) -> dict:
        agent_pos = self.last_event.metadata["agent"]
        countertop_info = self.get_object_by_id(countertop_id)
        counterttop_center = countertop_info["axisAlignedBoundingBox"]["center"]
        counterttop_size = countertop_info["axisAlignedBoundingBox"]["size"]
        c_max_x = counterttop_center["x"] + counterttop_size["x"] / 2
        c_min_x = counterttop_center["x"] - counterttop_size["x"] / 2
        c_max_z = counterttop_center["z"] + counterttop_size["z"] / 2
        c_min_z = counterttop_center["z"] - counterttop_size["z"] / 2
        # lower bound for random point targets - above the table
        c_max_y = counterttop_center["y"] + counterttop_size["y"] / 2

        hand_low = get_abs_obj1_from_rel_obj1_to_obj2(
            dict(
                position=RELATIVE_HAND_BOUNDS_DICT["low"], rotation=dict(x=0, y=0, z=0)
            ),
            agent_pos,
        )["position"]
        hand_high = get_abs_obj1_from_rel_obj1_to_obj2(
            dict(
                position=RELATIVE_HAND_BOUNDS_DICT["high"], rotation=dict(x=0, y=0, z=0)
            ),
            agent_pos,
        )["position"]
        if fixed_axis == "z":
            xlow = min(c_min_x, hand_low["x"]) + offset
            xhigh = max(c_max_x, hand_high["x"]) - offset
            zlow = max(c_min_z, hand_low["z"]) + offset
            zhigh = min(c_max_z, hand_high["z"]) - offset
        elif fixed_axis == "x":
            xlow = max(c_min_x, hand_low["x"]) + offset
            xhigh = min(c_max_x, hand_high["x"]) - offset
            zlow = min(c_min_z, hand_low["z"]) + offset
            zhigh = max(c_max_z, hand_high["z"]) - offset
        else:
            raise NotImplementedError(f"fixed axis {fixed_axis} not in ['x','z']")
        # above table, assuming the table height is smaller than arm limit
        ylow = c_max_y  # no need to add offset for height
        yhigh = hand_high["y"] - offset
        return dict(
            low=dict(x=xlow, y=ylow, z=zlow), high=dict(x=xhigh, y=yhigh, z=zhigh)
        )

    def find_closest_from_type(self, object_info: dict) -> dict:
        """name changed in dataset."""
        prev_object_id = object_info["object_id"]
        if self.get_object_by_id(prev_object_id, warning=False) is None:
            object_type = prev_object_id.split("|")[0]
            objects_of_type = self.get_object_by_type(object_type)
            if len(objects_of_type) > 1:
                get_logger().warning(
                    f"MULTIPLE OBJECTS: {object_type} in {self.controller.last_event.metadata['sceneName']}",
                )
            target_object = random.choice(objects_of_type)
            object_info["object_id"] = target_object["objectId"]
        return object_info

    def scale_object(
        self,
        object_id: str,
        scale: Union[int, float, list, tuple],
        scaleOverSeconds: int = 0,
        add_every_reset: bool = True,
    ):
        # as main camera for stretch robot is on the side
        force_action = object_id not in [
            obj["objectId"] for obj in self.visible_objects()
        ]
        scale_kwargs = dict(
            action="ScaleObject",
            objectId=object_id,
            scaleOverSeconds=scaleOverSeconds,
            forceAction=force_action,
        )
        if isinstance(scale, (int, float)):
            scale_kwargs["scale"] = scale
        else:
            assert isinstance(scale, Sequence) and len(scale) == 3
            get_logger().warning(f"risky as collision box maybe not expectedly scaled")
            scale_kwargs["scaleX"] = float(scale[0])
            scale_kwargs["scaleY"] = float(scale[1])
            scale_kwargs["scaleZ"] = float(scale[2])
        self.controller.step(scale_kwargs)
        self.wait()  # sim step for stable
        if add_every_reset:
            self.reset_extra_commands[f"Scale{object_id}"] = scale_kwargs
            self.reset_extra_commands[f"Scale{object_id}Wait"] = dict(
                action="AdvancePhysicsStep", simSeconds=1.0
            )

    def change_resolution(
        self, resolution: Union[tuple, list], add_every_reset: bool = False
    ):
        assert (
            isinstance(resolution, (tuple, list)) and len(resolution) == 2
        ), resolution
        if resolution == (
            self._start_player_screen_height,
            self._start_player_screen_width,
        ):
            return
        get_logger().debug(
            f"change resolution from "
            f"{self._start_player_screen_width, self._start_player_screen_height} "
            f"to {resolution}"
        )
        action_dict = {
            "action": "ChangeResolution",
            "x": resolution[0],
            "y": resolution[1],
        }
        self.controller.step(action_dict)
        if add_every_reset:
            self.reset_extra_commands["ChangeResolution"] = action_dict
            # attribute will not affect though as it reset to initial if not adding extra commands,
            # so only change if adding to every reset!
            (
                self._start_player_screen_width,
                self._start_player_screen_height,
            ) = resolution
        self.wait(2)  # need more for scaling up

    def wait(self, simSeconds: Union[int, float] = 1.0):
        self.controller.step(dict(action="AdvancePhysicsStep", simSeconds=simSeconds))

    @classmethod
    def correct_nan_inf(cls, flawed_dict: dict, extra_tag: str = "") -> dict:
        corrected_dict = copy.deepcopy(flawed_dict)
        anything_changed = 0
        for (k, v) in corrected_dict.items():
            if v != v or math.isinf(v):
                corrected_dict[k] = 0
                anything_changed += 1
        if anything_changed > 0:
            get_logger().warning(
                f"find and correct {anything_changed} nan/inf"
                + f" for {extra_tag}" * (len(extra_tag) > 0)
            )
        return corrected_dict

    @property
    def wrist_centric_frame(self) -> np.ndarray:
        frame = self.controller.last_event.third_party_camera_frames[1].copy()
        frame = remove_nan_inf_for_frames(frame, "wrist-centric")
        return frame

    @property
    def kinect_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        frame = self.controller.last_event.third_party_camera_frames[0].copy()
        frame = remove_nan_inf_for_frames(frame, "kinect_frame")
        # return kinect_reshape(frame)
        return frame

    @property
    def kinect_depth(self) -> np.ndarray:
        depth_frame = self.controller.last_event.third_party_depth_frames[0].copy()
        depth_frame = remove_nan_inf_for_frames(depth_frame, "depth_kinect")
        return kinect_reshape(depth_frame)

    @property
    def intel_frame(self) -> np.ndarray:
        """side camera, should not be used."""
        frame = self.controller.last_event.frame.copy()
        frame = remove_nan_inf_for_frames(frame, "intel_frame")
        return intel_reshape(frame)

    @property
    def intel_depth(self) -> np.ndarray:
        depth_frame = self.controller.last_event.depth_frame.copy()
        depth_frame = remove_nan_inf_for_frames(depth_frame, "depth_intel")
        return intel_reshape(depth_frame)

    @property
    def resolution(self) -> typing.Tuple[int, int]:
        return self._start_player_screen_height, self._start_player_screen_width

    def close(self):
        self.stop()

    def render(
        self, *, rgb_keys: Union[str, typing.Tuple[str, ...]], cat: bool = False
    ) -> Union[np.ndarray, typing.List[np.ndarray]]:
        if isinstance(rgb_keys, str):
            rgb_keys = [rgb_keys]
        frames = []
        for key in rgb_keys:
            if key in ["wrist", "wrist-centric", "wrist_centric"]:
                frames.append(self.wrist_centric_frame)
            elif key == "kinect":
                frames.append(self.kinect_frame)
            elif key == "intel":
                frames.append(self.intel_frame)
            elif key == "third-view":
                if key not in self.added_third_party_camera_info.keys():
                    camera_config = dict(
                        position=dict(x=-2, y=2.5035, z=2),
                        rotation=dict(x=36.885, y=-215.31, z=0),
                        fieldOfView=40,
                    )
                    self.add_third_party_camera(
                        name="third-view",
                        **camera_config,
                        skyboxColor="white",
                        add_every_reset=True,
                        update_if_existed=True,
                    )
                frames.append(
                    remove_nan_inf_for_frames(
                        self.controller.last_event.third_party_camera_frames[
                            self.added_third_party_camera_info[key]
                        ]
                    )
                )
            elif (
                self.added_third_party_camera_info is not None
                and key in self.added_third_party_camera_info.keys()
            ):
                frames.append(
                    remove_nan_inf_for_frames(
                        self.controller.last_event.third_party_camera_frames[
                            self.added_third_party_camera_info[key]
                        ]
                    )
                )
        if len(frames) > 1 and cat:
            return np.concatenate(frames, axis=1)
        if len(frames) == 1:
            return frames[0]
        return frames
