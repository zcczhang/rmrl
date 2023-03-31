from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, Literal

import gym
import gym.spaces
import mujoco_py
from gym.spaces import Box
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

from allenact.utils.system import get_logger
from allenact_plugins.sawyer_peg_plugin.constants import *
from allenact_plugins.utils import states_measure, states_entropy

_fastdtw_imported = True
try:
    from fastdtw import fastdtw
except ImportError:
    _fastdtw_imported = False


class MetaWorldV2Base(SawyerXYZEnv):
    """Base Env for MetaWorld manipulation tasks.

    Args:
        reset_free: whether reset free or episodic setting
        fix_phase_length: whether length of phases are fixed.
            False if considering done immediately if success/reach_max/irreversible
        almost_reversible: setting that hard reset immediately if reach the irreversible constraint, where
            the hand and object have not moved out of the thresholds for a number of consecutive phases
        tcp_std_irr_thresh: threshold for std of motion of the gripper
        obj_pos_std_irr_thresh: threshold for std of motion of the object (peg)
        num_phases_irr_tolerance: threshold for considering irreversible about the num of consecutive phases
            that the gripper and object not move out of the threshold
        num_steps_for_hard_resets: for two phases version, the number of fixed steps providing a fixed reset
        reward_type: name for the reward shaping implementing in `computer_reward()`
        compute_reward_kwargs: kwargs for `computer_reward()`
        obs_keys: keys for observation space, low-dim state and/or rgb image
        state_keys: keys for low-dim state obs, "all" if ("tcp_center", "tcp_dist", "obj_pos", "target")
        rgb_views: view for rgb image obs, all for 3rd person view and (wrist) hand-centric view, or other camera view
        rgb_resolution: rgb obs image size
    """

    max_path_length = 300  # for switching phases
    TARGET_RADIUS = 0.05  # 0.07

    def __init__(
        self,
        *,
        # general setting
        reset_free: bool = False,
        fix_phase_length: bool = False,
        # reset free setting: simple way to detect near-irreversible / reset every fixed long horizon
        almost_reversible: bool = False,
        state_measure_config: Union[
            Literal["std", "euclidean", "dtw", "entropy"], dict
        ] = "std",
        num_steps_for_hard_resets: Optional[Union[int, float]] = np.inf,
        reset_if_explicit_irreversible: bool = False,
        # reward config
        reward_type: str = "navigation",
        compute_reward_kwargs: Optional[dict] = None,
        # observation space (state and/or rgb obs)
        obs_keys: Union[str, tuple] = ("rgb", "state"),
        state_keys: Optional[Union[str, tuple]] = "all",
        rgb_views: Optional[Union[str, tuple]] = ("3rd", "hand_centric"),
        rgb_resolution: tuple = (84, 84),
        **kwargs,
    ):
        hand_low = (-0.6, 0.25, 0.045)
        hand_high = (0.6, 1.1, 0.3)
        assert obs_keys, "at least one obs key has to be specified"
        super().__init__(
            self.model_name, hand_low=hand_low, hand_high=hand_high, **kwargs
        )
        assert self.isV2

        self.reversible_space = Box(
            np.array([hand_low[0], hand_low[1], -0.05]), np.array(hand_high)
        )

        # For override
        self.goal_space = None
        self.initial_states = None
        self.goal_states = None
        self.obj_init_pos = None
        self.hand_init_pos = None
        self.peg_head_pos_init = None
        self.goal = None
        self._target_pos = None

        self._partially_observable = False
        self._set_task_called = True  # just pass this as True
        self._freeze_rand_vec = False
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self._reward_type = reward_type
        self.compute_reward_kwargs = compute_reward_kwargs or {}
        # for logging metrics
        self.info = {}
        self.is_irreversible = 0.0
        # terminal reward
        self.global_reward = 10
        # done
        self.done = False
        # False if considering done immediately if success/reach_max/irreversible
        self.fix_phase_length = fix_phase_length
        self.num_hard_resets = 0.0
        self.cum_env_steps = 0.0
        self.num_steps_for_hard_resets = num_steps_for_hard_resets or np.inf
        self.cum_steps_for_reset = 0  # for fixed num of hard reset
        self.hard_reset = True
        self.reset_if_explicit_irreversible = reset_if_explicit_irreversible

        # Reset Free specific config
        self.reset_free = reset_free
        self.almost_reversible = almost_reversible
        if almost_reversible:
            # not provide hard reset every fixed steps
            self.num_steps_for_hard_resets = np.inf
            self.num_phases_irr = 0  # consecutive near-irr phases
            if isinstance(state_measure_config, str):
                if state_measure_config == "dtw":
                    if not _fastdtw_imported:
                        raise ImportError(
                            "require to install fastdtw for calculating dtw: `pip install fastdtw`"
                        )
                self.state_measure = STATE_MEASURE_CONFIGS[state_measure_config]
            else:
                self.state_measure = state_measure_config
            assert (
                key in self.state_measure
                for key in ["history", "thresholds", "metrics"]
            )

        if obs_keys == "all":
            obs_keys = ("rgb", "state")
        if state_keys == "all":
            state_keys = ("tcp_center", "tcp_dist", "obj_pos", "target")
        if rgb_views == "all":
            rgb_views = ("3rd", "hand_centric")
        if isinstance(obs_keys, str):
            obs_keys = (obs_keys,)
        if isinstance(state_keys, (str, type(None))):
            state_keys = (state_keys,)
        if isinstance(rgb_views, (str, type(None))):
            rgb_views = (rgb_views,)
        if "rgb" in obs_keys:
            assert rgb_views[0] is not None
        if "state" in obs_keys:
            assert state_keys[0] is not None
        self.obs_keys = obs_keys
        self.state_keys = state_keys
        self.rgb_views = rgb_views
        self.rgb_resolution = rgb_resolution
        self._render_gpu_id = None  # -1 for default first available GPU

    @property
    def model_name(self):
        raise NotImplementedError

    def _rgb_obs_space(self) -> gym.spaces.Box:
        """concatenate rgb obs, otherwise specify sensors for each view."""
        return gym.spaces.Box(
            0,
            255,
            [3]
            + [self.rgb_resolution[0], self.rgb_resolution[1] * len(self.rgb_views)],
            np.uint8,
        )

    def _state_obs_space(self) -> Union[gym.spaces.Box, None]:
        if "state" not in self.obs_keys:
            return None
        state_space_low = []
        state_space_high = []
        for key in self.state_keys:
            if key == "tcp_center":
                state_space_low.append(self._HAND_SPACE.low)
                state_space_high.append(self._HAND_SPACE.high)
            elif key == "tcp_dist":
                state_space_low.append(-1)
                state_space_high.append(1)
            elif key == "obj_pos":
                state_space_low.append(np.full(3, -np.inf))
                state_space_high.append(np.full(3, +np.inf))
            elif key == "target":
                assert self.goal_space is not None
                state_space_low.append(self.goal_space.low)
                state_space_high.append(self.goal_space.high)
            else:
                raise ValueError(f"state key {key} is not supported!")
        return gym.spaces.Box(np.hstack(state_space_low), np.hstack(state_space_high))

    @property
    def observation_space(self) -> Union[gym.spaces.Box, gym.spaces.Dict]:
        """Box space if only low-dim state is obs, otherwise Dict space with
        key `rgb` and `state`"""
        if "rgb" not in self.obs_keys:
            return self._state_obs_space()
        state_space = self._state_obs_space()
        return (
            gym.spaces.Dict(rgb=self._rgb_obs_space(), state=state_space)
            if state_space is not None
            else gym.spaces.Dict(rgb=self._rgb_obs_space())
        )

    @property
    def obj_pos(self) -> np.ndarray:
        """in case retrieve somewhere else (e.g. point cloud viz)"""
        return self._get_pos_objects()

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        """10 dimension (full-state) observation center pos of grippers (3,),
        gripper distance (1,), pegHead pos (3,), target pos (3,)"""
        obs = super()._get_obs()
        # xyz and gripper distance for end effector
        tcp_center = self.tcp_center
        tcp_dist = np.array([obs[3]])
        obj_head = self._get_pos_objects()
        flatten_obs = np.concatenate([tcp_center, tcp_dist, obj_head, self._target_pos])
        return OrderedDict(
            obs=flatten_obs,
            tcp_center=tcp_center,
            tcp_dist=tcp_dist,
            obj_pos=obj_head,
            target=self._target_pos,
        )

    def _get_flatten_full_obs(self) -> np.ndarray:
        return self._get_obs_dict()["obs"]

    def _get_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = dict()
        for obs_key in self.obs_keys:
            if obs_key == "state":
                obs_dict = self._get_obs_dict()
                assert self.state_keys is not None
                assert set(self.state_keys).issubset(set(obs_dict.keys()))
                obs[obs_key] = np.concatenate(
                    [v for k, v in obs_dict.items() if k in self.state_keys]
                )
            elif obs_key == "rgb":
                obs[obs_key] = self.get_vis_obs()
        return obs if "rgb" in self.obs_keys else obs["state"]

    def get_vis_obs(
        self, resolution: Optional[tuple] = None, transpose: bool = True
    ) -> np.ndarray:
        rgb = []
        resolution = resolution or self.rgb_resolution
        for view in self.rgb_views:
            rgb.append(self.offscrean_render(view, resolution, transpose=False))
        rgb = np.concatenate(rgb, axis=1)
        if transpose:
            rgb = rgb.transpose([2, 0, 1])
        return rgb

    def offscrean_render(
        self, view: str, resolution: Optional[tuple] = None, transpose: bool = True
    ) -> np.ndarray:
        """could get rgb obs from exp sensors outside of the env."""
        if view == "3rd":
            camera_name = "rgbobs"  # view_3
        elif view == "hand_centric":
            camera_name = "view_1"
        else:
            camera_name = view
        img = self.sim.render(*resolution, mode="offscreen", camera_name=camera_name)
        if transpose:
            img = img.transpose([2, 0, 1])
        return img

    def get_next_goal(self) -> np.ndarray:
        raise NotImplementedError

    def reset_goal(self, goal: Optional[np.ndarray] = None):
        goal = goal or self.get_next_goal()
        self.goal = goal
        self._target_pos = goal[4:]
        self.sim.model.site_pos[self.model.site_name2id("goal")] = self._target_pos

    @_assert_task_is_set
    def step(self, action: Union[np.ndarray, list]) -> tuple:
        if isinstance(action, list):
            action = np.array(action)
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        self.curr_path_length += 1
        self.cum_steps_for_reset += 1
        self.cum_env_steps += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            self.info = {"irreversible_rate": 1.0}
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                True,  # reset if irreversible
                {},
            )

        self._last_stable_obs = self._get_obs()

        reward, self.info = self.evaluate_state(self._last_stable_obs, action)
        self.info["num_hard_resets"] = self.num_hard_resets

        self.done = self.curr_path_length >= self.max_path_length
        self.done = (
            self.done
            if self.fix_phase_length
            else (self.done or bool(self.info["success"]) or bool(self.is_irreversible))
        )

        if self.almost_reversible:
            obs_dict = (
                self._get_obs_dict()
            )  # in case raw rgb obs only that not get obs_dict before
            history = self.state_measure["history"]
            metrics = self.state_measure["metrics"]
            measure_name = metrics["name"]
            for key in history.keys():
                assert key in metrics, f"{key} not in {metrics.keys()}"
                if key in obs_dict:
                    self.state_measure["history"][key].append(obs_dict[key])
                elif key == "tcp_and_obj":
                    self.state_measure["history"][key].append(
                        np.concatenate([obs_dict["tcp_center"], obs_dict["obj_pos"]])
                    )
                else:
                    raise NotImplementedError(key)
                if metrics[key] > 0:
                    self.info[f"{key}_{measure_name}"] = metrics[key]

            self.info["num_phases_irr"] = self.num_phases_irr

        return self._last_stable_obs, reward, self.done, {}

    @_assert_task_is_set
    def evaluate_state(
        self, obs: np.ndarray, action: np.ndarray, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """return the reward and info for the current step."""
        return self.compute_reward(obs, action, **self.compute_reward_kwargs, **kwargs)

    def _get_pos_objects(self):
        raise NotImplementedError

    def _get_quat_objects(self):
        raise NotImplementedError

    def compute_reward(
        self, obs: np.ndarray, action: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        raise NotImplementedError

    def _fake_reset_model(self) -> np.ndarray:
        # a fake reset for reset free version
        self.reset_goal()
        self.stable_rebuild_obj()
        return self._get_obs()

    def reset_model(self):
        raise NotImplementedError

    def stable_rebuild_obj(self):
        """rebuild the obj at its original position to prevent environment
        instability."""
        self._set_obj_xyz(self.get_env_state()[0][1][9:12])
        # fake episode-wise obj_init_pos
        self.obj_init_pos = self.get_env_state()[0][1][9:12]
        self.peg_head_pos_init = self._get_pos_objects()

    def _reset_measure_history(self):
        for k in self.state_measure["history"].keys():
            self.state_measure["history"][k] = []

    def measure_irreversible(self, reset_history: bool = True):
        measure_name = self.state_measure["metrics"]["name"]
        history = self.state_measure["history"]
        thresholds = self.state_measure["thresholds"]
        if measure_name == "std" or "entropy" in measure_name:
            irr = self.dispersion_measure_check(measure_name, history, thresholds)
        elif measure_name in ["euclidean", "dtw"]:
            irr = self.distance_measure_check(measure_name, history, thresholds)
            reset_history = False  # reset inside based on memory & measure steps
        else:
            raise NotImplementedError(measure_name)
        if reset_history:
            # by default history reset every phase
            self._reset_measure_history()
        return irr

    def dispersion_measure_check(
        self, method: str, history: dict, thresholds: dict
    ) -> bool:
        """simply measured by phi(state), will phi could be std, entropy, or
        any other function to measure the dispersion within each phase/time
        horizon."""
        irr = False
        _is_irr = []
        for key, key_history in history.items():
            if len(key_history) > 0:  # maybe longer
                if method == "std":
                    k_measure = states_measure(
                        key_history, metrics="std", mean_xyz=True
                    )
                elif "entropy" in method:
                    grid_size = self.state_measure["grid_size"]
                    k_measure = states_entropy(
                        key_history,
                        grid_size=grid_size,
                        world_space=self.reversible_space,
                    )
                else:
                    raise NotImplementedError(method)
                # update logging metrics
                self.state_measure["metrics"][key] = k_measure
                _is_irr.append(0 < k_measure < thresholds[key])
        if 0 < len(_is_irr) <= sum(_is_irr):  # i.e. all pass threshold for irreversible
            self.num_phases_irr += 1
        if self.info.get("success", False):
            self.num_phases_irr = 0
        if self.num_phases_irr >= thresholds["num_phases_irr_tolerance"]:
            irr = True
            self.num_phases_irr = 0
        return irr

    def distance_measure_check(
        self, method: str, history: dict, thresholds: dict
    ) -> bool:
        """max(D(p_{m+i}, {p_0, ..., p_{m-n+i})), m > n, i from 0 to n
        where D(p_{m+i}, {p_0, ..., p_{m-n+i}) = min(d(p_{m+i}, p_j)), j from 0 to m-n+i,
        m is `memory` taking from previous steps, and n is the measure steps that taking after the memory
        p_i is the state at timestep i where p_0 indicates the start of the memory,
        d is distance measure (Euclidean, dynamic time warping (DTW), or any others),
        """
        if self.info.get("success", False):
            self._reset_measure_history()
            return False
        time_horizons = self.state_measure["time_horizons"]
        memory, measure_steps = time_horizons["memory"], time_horizons["measure_steps"]
        if "schedule" in self.state_measure:
            # just provide example for scheduling the horizon but not necessary
            min_step, max_step, schedule_steps = self.state_measure["schedule"].values()
            if min_step < max_step and self.cum_env_steps <= schedule_steps != 0:
                memory = min_step + int(
                    (max_step - min_step) * min(1, self.cum_env_steps / schedule_steps)
                )

        nsteps = -1  # placeholder
        irr = False
        _is_irr = []
        for key, key_history in history.items():
            if len(key_history) <= 0:
                continue
            state = np.stack(key_history, axis=0)
            nsteps = state.shape[0]
            if nsteps < memory + measure_steps:
                return False
            distances = []
            for i in range(measure_steps):
                p = key_history[memory + i]
                past_states = key_history[: memory - measure_steps + 1]
                if method == "euclidean":
                    min_distance_to_past_states = np.min(
                        np.linalg.norm(p - past_states, axis=-1)
                    )
                elif method == "dtw":
                    distances_ = []
                    for past_state in past_states:
                        distance, _ = fastdtw(p, past_state)
                        distances_.append(distance)
                    # min_distance_to_past_states = np.mean(distances_)
                    min_distance_to_past_states = np.min(distances_)
                else:
                    raise NotImplementedError(method)
                distances.append(min_distance_to_past_states)
            maxmin_dist = np.max(distances)
            # NOTE: logging will be every `memory` + `measure_steps` for the last previous instead
            self.state_measure["metrics"][key] = maxmin_dist
            _is_irr.append(0 < maxmin_dist < thresholds[key])
        if 0 < len(_is_irr) <= sum(_is_irr):  # i.e. all pass threshold for irreversible
            self.num_phases_irr += 1
            irr = True  # check every memory + measure_steps, which is large enough
        if nsteps >= memory + measure_steps:
            self._reset_measure_history()
        return irr

    def reset(self) -> np.ndarray:
        if self.almost_reversible:
            if self.measure_irreversible():
                self.is_irreversible = True
                self.hard_reset = True

        self.curr_path_length = 0
        self.done = False
        # for provide hard reset every fixed number of steps
        if (
            self.reset_free
            and not self.almost_reversible  # though not necessary
            and self.cum_steps_for_reset >= self.num_steps_for_hard_resets
        ):
            # cum_steps_for_reset will be reset to 0 at super reset()
            self.hard_reset = True
            self.cum_steps_for_reset = 0
        # hard reset when irreversible
        if self.is_irreversible or self.hard_reset:
            # self.is_irreversible is updated step-wise
            self.hard_reset = True
            self._did_see_sim_exception = False
            self.num_hard_resets += 1
            self.cum_steps_for_reset = 0
            return super().reset()
        return self.reset_model()

    def is_successful(self, obs: Optional[np.ndarray] = None) -> bool:
        raise NotImplementedError

    def setup_render_gpu_id(self, gpu_id: Optional[int]):
        """fn calling outside the cls after env initializes for distributed
        training (e.g. task sampler), instead of automatically setting as in
        case env is initialized by gym.make(`str`)"""
        gpu_id = gpu_id or self._render_gpu_id
        if len(self.sim.render_contexts) == 0 and gpu_id is not None:
            if "rgb" not in self.obs_keys:
                get_logger().warning(
                    f"trying to setup gpu_id though env obs space doesn't contain rgb. "
                    f"Only reasonable if using Sensor rendered rgb obs outside of the env"
                )
            try:
                # set only once
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, device_id=gpu_id
                )
                self._render_gpu_id = gpu_id
            except RuntimeError as e:
                if gpu_id != -1:
                    get_logger().warning(
                        f"WARNING: gpu_id {gpu_id} is not available: {e}"
                    )
                else:
                    raise RuntimeError(e)
        elif gpu_id is not None:
            get_logger().warning(
                f"WARNING: gpu id can only set once so far. Current: {self._render_gpu_id}"
            )
