import copy
from typing import Optional, Union, Tuple, Dict, Literal

import gym
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from allenact_plugins.sawyer_peg_plugin.base import MetaWorldV2Base
from allenact_plugins.sawyer_peg_plugin.constants import *
from allenact_plugins.utils import (
    get_file,
    stable_mul,
)

__all__ = ["SawyerPegV2Base"]


class SawyerPegV2Base(MetaWorldV2Base):
    """SawyerPeg env.

    Args:
        reset_free: whether reset free or episodic setting
        random_targets: sample random targets for each phase of reset free
        peg_insert_goal_prob: the probability that sampled target is just insert peg for random targets setting
        two_phases: adversarial plug and unplug setting for reset free
        small_table: using smaller table for MuJuCo model
        random_init_box: whether random initialize peg box each (hard) reset
        random_box_hole: whether sampling random box hole within the box each (hard) reset
        obj_goal_space_type: basic or wider object and goal sample space (different training and eval distribution)
        others see MetaWorldV2Base
    """

    def __init__(
        self,
        *,
        # if reset free, either random target or 2 phases adv
        reset_free: bool = False,
        random_targets: bool = False,
        peg_insert_goal_prob: float = 0.3,
        two_phases: bool = False,
        small_table: bool = False,
        random_init_box: bool = True,
        random_box_hole: bool = False,
        obj_goal_space_type: Literal["basic", "wider"] = "basic",
        almost_reversible: bool = False,
        state_measure_config: Union[str, dict] = "std",
        num_steps_for_hard_resets: Optional[Union[int, float]] = np.inf,
        reset_if_explicit_irreversible: bool = False,
        penalty_if_explicit_irreversible: Union[int, float] = 0.0,
        reward_type: Literal["navigation", "sparse"] = "navigation",
        compute_reward_kwargs: Optional[Union[dict, Literal["full", "min"]]] = "full",
        fix_phase_length: bool = False,
        obs_keys: Union[Literal["all", "rgb", "state"], Tuple[str, ...]] = "state",
        state_keys: Optional[Union[str, Tuple[str, ...]]] = "all",
        rgb_views: Optional[
            Union[Literal["all", "3rd", "hand_centric"], Tuple[str, ...]]
        ] = ("3rd", "hand_centric"),
        rgb_resolution: tuple = (84, 84),
    ):
        # to build MuJoCo model
        space_dict = BASIC_SPACE if obj_goal_space_type == "basic" else WIDER_SPACE
        obj_low, obj_high = space_dict["obj_low"], space_dict["obj_high"]
        goal_low, goal_high = space_dict["goal_low"], space_dict["goal_high"]
        self.small_table = small_table
        if isinstance(compute_reward_kwargs, str):
            compute_reward_kwargs = POSSIBLE_REWARD_KWARGS[compute_reward_kwargs]
        if random_targets:
            assert not random_box_hole and not random_init_box
        super().__init__(
            reset_free=reset_free,
            fix_phase_length=fix_phase_length,
            almost_reversible=almost_reversible,
            state_measure_config=state_measure_config,
            num_steps_for_hard_resets=num_steps_for_hard_resets,
            reset_if_explicit_irreversible=reset_if_explicit_irreversible,
            reward_type=reward_type,
            compute_reward_kwargs=compute_reward_kwargs,
            obs_keys=obs_keys,
            state_keys=state_keys,
            rgb_views=rgb_views,
            rgb_resolution=rgb_resolution,
            # **kwargs,
        )
        self.init_config = {
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.initial_states = copy.deepcopy(INITIAL_STATES)
        self.goal_states = copy.deepcopy(GOAL_STATES)

        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        self.goal = copy.deepcopy(GOAL_STATES[0])
        self._target_pos = self.goal[4:]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)), np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0.03, 0.0, 0.13]),
            np.array(goal_high) + np.array([0.03, 0.0, 0.13]),
        )

        if reward_type == "navigation":
            self.prev_obj_to_target = -1
            self.prev_gripper_dist = -1
            self.prev_hand_to_peg = -1
            self.prev_peg_z = -1
            self.peg_first_grasped = False
        assert penalty_if_explicit_irreversible <= 0
        self.penalty_if_explicit_irreversible = penalty_if_explicit_irreversible

        if reset_free:
            assert two_phases or random_targets
        if two_phases:
            assert not random_targets
            # Alice (0) or Bob (1) phase
            self.phase = 0
            # task successes
            self.phase0_success = False
            self.phase1_success = False
            # for overall success
            self.success_overall = False
        elif random_targets:
            self.random_target_space = (
                gym.spaces.Box(np.array([-0.3, 0.4, 0.02]), np.array([0.15, 0.8, 0.13]))
                if not self.small_table
                else gym.spaces.Box(
                    np.array([-0.3, 0.5, 0.02]), np.array([0.15, 0.7, 0.13])
                )
            )  # make sure the target is on top of the table (narrower than the first)

            # just sample once fixed box there
            self.pos_box = None
            # initial reset for sampling a fixed target box
            self.peg_insert_goal_prob = peg_insert_goal_prob
            self.peg_insert_goal = None

        self.random_targets = random_targets
        self.two_phases = two_phases
        self.init_reset = True

        self.random_init_box = random_init_box  # random box pos
        # save the original arg in case if there's hard reset every 1e4 or 1e5 steps
        self._random_init_box_every_reset = self.random_init_box
        self.random_box_hole = random_box_hole  # random hole pos within box
        if random_box_hole:
            self.peg_box_base_sizes = None
            self.peg_box_base_poses = None

    def _episodic_goal(self) -> np.ndarray:
        """episodic.

        Just follow the EARL impl, though only the target_pos of the
        goal is used
        """
        num_goals = self.goal_states.shape[0]
        goal_idx = self.np_random.randint(0, num_goals)
        next_goal = self.goal_states[goal_idx]
        if self.random_init_box:
            next_goal[-3:] = self.goal_space.sample()
        return next_goal

    def get_next_goal(self) -> np.ndarray:
        if self.two_phases:
            if self.phase == 0:
                # Alice: plug
                return self._episodic_goal()
            else:
                # Bob: unplug (backward)
                num_goals = self.initial_states.shape[0]
                goal_idx = self.np_random.randint(0, num_goals)
                return self.initial_states[goal_idx].copy()
        elif self.random_targets:
            if self.init_reset:
                self.init_reset = False
                goal = self._episodic_goal()
                self.peg_insert_goal = copy.deepcopy(goal)
            elif self.np_random.random() > self.peg_insert_goal_prob:
                goal = self.get_a_random_goal(self._episodic_goal())
            else:
                goal = copy.deepcopy(self.peg_insert_goal)
            return goal
        return self._episodic_goal()

    def _fully_reset_model(self, reset_hand: bool = True) -> np.ndarray:
        """reset model completely in episodic (hard reset) manner.

        Pipeline:
            - reset hand pos if `reset_hand`
            - reset goal (get next goal in terms of the task seting (random, two-phases, or simple episodic))
            - reset peg box (random if random_init_box, or rebuild box if it's fixed)
            - reset peg (depends on phases if `two_phases`, or random on the table)
            - reset hole of the box if `random_box_hole`, and update target and goal accordingly
        """
        if reset_hand:
            self._reset_hand()
        self.reset_goal()
        # (re-)build peg box
        if self.random_targets:
            # fixed pos box all the time
            if self.pos_box is None:
                self.pos_box = self.goal_states[0][4:] - np.array([0.03, 0.0, 0.13])
            pos_box = self.pos_box
        else:
            # potentials random init peg box
            pos_box = self.goal_states[0][4:] - np.array([0.03, 0.0, 0.13])

        self.sim.model.body_pos[self.model.body_name2id("box")] = pos_box
        # reset peg obj
        # note that for reset free version, it is always reset at the table but not the goal
        if (
            not self.two_phases
            or self.phase == 0
            or (self.two_phases and self.reset_free)
        ):
            pos_peg = self.obj_init_pos
            if self.random_init:
                pos_peg, _ = np.split(self._get_state_rand_vec(), 2)
                while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
                    pos_peg, _ = np.split(self._get_state_rand_vec(), 2)
            self.obj_init_pos = pos_peg
        else:
            goal_pos = self.goal_states[0][4:] - np.array([-0.1, 0.0, 0.0])
            self.obj_init_pos = goal_pos + self.np_random.uniform(-0.02, 0.02, size=3)
        self._set_obj_xyz(self.obj_init_pos)
        self.peg_head_pos_init = self._get_site_pos("pegHead")
        # random hole (and will update target)
        if self.random_box_hole:
            self.change_peg_hole()
        return self._get_obs()

    def reset_model(self) -> np.ndarray:
        if not self.reset_free or self.init_reset or self.hard_reset:
            if self.reset_free and (self.hard_reset or self.init_reset):
                self.hard_reset = False
                # re-assign the random_init_box arg
                if not self.random_targets:
                    self.init_reset = (
                        False  # flip after setting the first goal for random target
                    )
                    self.random_init_box = self._random_init_box_every_reset
            return self._fully_reset_model(reset_hand=True)
        else:
            if not self.hard_reset and self.reset_free:
                # only randomly sample box position at beginning of hard reset in reset free training
                self.random_init_box = False
            # reset free
            return self._fake_reset_model()

    def reset(self) -> np.ndarray:
        if self._reward_type == "navigation":
            self.prev_obj_to_target = -1
            self.peg_first_grasped = False
            if not self.reset_free:
                # for reset free version, the below commented dists should not be reset
                self.prev_gripper_dist = -1
                self.prev_hand_to_peg = -1
                self.prev_peg_z = -1
        if self.two_phases:
            # phase 0 success reset in the TaskSampler
            self.phase1_success = False
            self.success_overall = False
            # self.phase = 1 - self.phase   # switch in TaskSampler
        return super().reset()

    @property
    def model_name(self):
        return (
            get_file(
                __file__,
                "sawyer_peg_assets/sawyer_xyz",
                "sawyer_peg_insertion_side_small_table.xml",
            )
            if self.small_table
            else get_file(
                __file__,
                "sawyer_peg_assets/sawyer_xyz",
                "sawyer_peg_insertion_side.xml",
            )
        )

    def _get_pos_objects(self):
        return self._get_site_pos("pegHead")

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_site_xmat("pegHead")).as_quat()

    def get_a_random_goal(self, goal: Optional[np.ndarray] = None) -> np.ndarray:
        assert self.random_targets, "only used for random targets env variant"
        goal = goal if goal is not None else self.get_next_goal()
        # sample random target in space
        while True:
            random_target = self.random_target_space.sample()
            goal[-3:] = random_target
            if self.check_target_not_in_box(
                random_target
            ) and self.reversible_space.contains(random_target):
                break
        return goal

    def check_target_not_in_box(
        self, target_pos: np.ndarray, thresh: float = 0.1
    ) -> bool:
        # prevent the goal inside the box
        tlc_col_box_1 = self._get_site_pos("top_left_corner_collision_box_1")
        brc_col_box_1 = self._get_site_pos("bottom_right_corner_collision_box_1")
        x, y, z = target_pos
        # not inside or behind the box
        inside_box = (
            # positive y
            tlc_col_box_1[1] - thresh / 2
            <= y
            <= brc_col_box_1[1] + thresh / 2
        ) and (
            x <= brc_col_box_1[0] + thresh
        )  # negative x
        return not inside_box

    def change_peg_hole(self):
        # init, record the original geom size and pos
        init = False
        if self.peg_box_base_sizes is None:
            self.peg_box_base_sizes = dict()
            self.peg_box_base_poses = dict()
            self.peg_box_base_poses["hole"] = copy.deepcopy(
                self.sim.model.site_pos[self.model.site_name2id("hole")]
            )
            init = True
        ids = []
        for name in [
            "below_of_hole",
            "left_of_hole",
            "right_of_hole",
            "top_of_hole",
        ]:
            _id = self.model.geom_name2id(name)
            ids.append(_id)
            if init:
                self.peg_box_base_sizes[name] = copy.deepcopy(
                    self.sim.model.geom_size[_id]
                )
                self.peg_box_base_poses[name] = copy.deepcopy(
                    self.sim.model.geom_pos[_id]
                )
            else:
                self.sim.model.geom_size[_id] = self.peg_box_base_sizes[name]
                self.sim.model.geom_pos[_id] = self.peg_box_base_poses[name]
        below_id, left_id, right_id, top_id = ids

        z_offset = self.np_random.uniform(low=-0.015, high=0.05)
        # modify the size and pos of the bottem box of the hole
        self.sim.model.geom_size[below_id][2] -= z_offset
        self.sim.model.geom_pos[below_id][2] *= (
            self.sim.model.geom_size[below_id][2]
            / self.peg_box_base_sizes["below_of_hole"][2]
        )
        # self.sim.model.geom_pos[below_id][2] += 0.0

        # modify the size and pos of the top box of the hole
        self.sim.model.geom_size[top_id][2] += z_offset
        self.sim.model.geom_pos[top_id][2] *= (
            1
            - self.sim.model.geom_size[top_id][2]
            / self.peg_box_base_poses["top_of_hole"][2]
        )
        self.sim.model.geom_pos[top_id][2] += 0.015

        # corresponding target, and right and left box vertical offset
        target_z_change = (
            self.peg_box_base_poses["top_of_hole"][2]
            - self.sim.model.geom_pos[top_id][2]
        ) + (
            self.peg_box_base_poses["below_of_hole"][2]
            - self.sim.model.geom_pos[below_id][2]
        )
        for name in [left_id, right_id]:
            self.sim.model.geom_pos[name][2] -= target_z_change

        horizontal_offset = self.np_random.uniform(low=-0.03, high=0.03)
        # modify the size and pos of the left box of the hole
        self.sim.model.geom_size[left_id][0] -= horizontal_offset
        self.sim.model.geom_pos[left_id][0] *= (
            self.sim.model.geom_size[left_id][0]
            / self.peg_box_base_sizes["left_of_hole"][0]
        )
        self.sim.model.geom_pos[left_id][0] -= 0.03 * horizontal_offset / 0.01

        # modify the size and pos of the right box of the hole
        self.sim.model.geom_size[right_id][0] += horizontal_offset
        self.sim.model.geom_pos[right_id][0] *= (
            1
            - self.sim.model.geom_size[right_id][0]
            / self.peg_box_base_poses["right_of_hole"][0]
        )
        self.sim.model.geom_pos[right_id][0] += 0.03

        # corresponding target vertical offset
        _target_pos = copy.deepcopy(self._target_pos)
        _target_pos[-1] -= target_z_change
        target_horizontal_change = (
            self.peg_box_base_poses["right_of_hole"][0]
            - self.sim.model.geom_pos[right_id][0]
        ) + (
            self.peg_box_base_poses["left_of_hole"][0]
            - self.sim.model.geom_pos[left_id][0]
        )
        _target_pos[1] -= target_horizontal_change

        # hole for visual only
        self.sim.model.site_pos[self.model.site_name2id("hole")] = (
            self.peg_box_base_poses["hole"][0] - target_horizontal_change,
            self.peg_box_base_poses["hole"][1],
            self.peg_box_base_poses["hole"][2] - target_z_change,
        )
        # update goal
        self.goal[-3:] = _target_pos
        self._target_pos = _target_pos
        self.sim.model.site_pos[self.model.site_name2id("goal")] = self._target_pos

        # make the box building stable
        for _ in range(10):
            self.sim.step()

    def compute_reward(
        self,
        obs: Union[np.ndarray, dict],
        action: Optional[np.ndarray] = None,
        *,
        obj_to_target_reward_coef: Union[int, float] = 100.0,
        tcp_to_obj_reward_coef: Union[int, float] = 100.0,
        gripper_width_reward_coef: Union[int, float] = 2.0,
        z_reward_coef: Union[int, float] = 2.0,
        use_z_reward: bool = True,
        obj_to_target_scale: Union[np.ndarray, list, float, int] = np.array([1, 2, 2]),
        use_first_grasped_reward: bool = True,
        use_gripper_reward: bool = True,
    ) -> Tuple[float, Dict[str, float]]:
        obs = self._get_flatten_full_obs()  # in case partial state obs space
        # pos of gripper center
        tcp = obs[:3]
        # pegGrasp pos
        obj = self._get_site_pos("pegGrasp")
        # pegHead pos
        obj_head = obs[4:7]
        # gripper distance
        tcp_opened = obs[3]
        target = obs[-3:]
        # gripper center to pegGrasp distance
        tcp_to_obj = np.linalg.norm(obj - tcp)

        # pegHead to target pos "scaled" distance if `obj_to_target_scale` is different from [1, 1, 1]
        # force agent to pick up object then insert (weight more y, z)
        obj_to_target = np.linalg.norm(
            (obj_head - target) * np.array(obj_to_target_scale)
        )

        # terminal reward
        success = float(self.is_successful(obs))
        terminal_reward = self.global_reward if success else 0
        self.is_irreversible = float(self.is_irreversible_state())
        irr_penalty = (
            self.penalty_if_explicit_irreversible * int(self.is_irreversible)
            if self.reset_if_explicit_irreversible
            else 0.0
        )

        if self._reward_type == "sparse":
            return (
                terminal_reward + irr_penalty,
                dict(
                    step_reward=terminal_reward,
                    success=success,
                    tcp_to_obj=tcp_to_obj,
                    # tcp_to_obj_reward=0,
                    # gripper_reward=0,
                    obj_to_target=obj_to_target,
                    # obj_to_target_reward=0,
                    # z_reward=0,
                    irrevresible=self.is_irreversible,
                ),
            )
        elif self._reward_type == "navigation":
            if self.prev_obj_to_target < 0:
                # init
                self.prev_obj_to_target = obj_to_target
                self.prev_gripper_dist = tcp_opened
                self.prev_hand_to_peg = tcp_to_obj
                self.prev_peg_z = obj_head[2]

            # minimal distance-diff-based reward shaping based on gripper, obj, and target
            # hand to peg reward
            tcp_to_obj_reward = stable_mul(
                self.prev_hand_to_peg - tcp_to_obj, tcp_to_obj_reward_coef
            )
            self.prev_hand_to_peg = tcp_to_obj
            # dist-based reward from peg to goal,
            # positive if closer, negative if further
            obj_to_target_reward = stable_mul(
                self.prev_obj_to_target - obj_to_target, obj_to_target_reward_coef
            )
            self.prev_obj_to_target = obj_to_target

            # Other distance-diff-based reward shaping that maybe helpful but not used in paper

            # grasp reward: gripper as close as possible if on the peg,
            # reverse if not on the peg (as open as possible)
            if not use_gripper_reward:
                gripper_reward = 0
            elif tcp_to_obj < self.TARGET_RADIUS:
                # as closed as possible
                gripper_reward = stable_mul(
                    self.prev_gripper_dist - tcp_opened, gripper_width_reward_coef
                )
                if (
                    use_first_grasped_reward and obj_head[2] > self.obj_init_pos[2]
                ):  # and contact:
                    # only reward the first time successfully grasped
                    if not self.peg_first_grasped:
                        gripper_reward += 1
                        self.peg_first_grasped = True
            else:
                # as open as possible
                gripper_reward = stable_mul(
                    tcp_opened - self.prev_gripper_dist, gripper_width_reward_coef
                )
                # # this penalty hurts
                # if self.peg_first_grasped:
                #     gripper_reward -= 1
            self.prev_gripper_dist = tcp_opened

            # z axis condition
            z_reward = 0
            if use_z_reward:
                peg_z = obj_head[2]
                target_z = target[2]
                if peg_z > target_z:
                    # curr should < prev
                    z_reward = stable_mul((self.prev_peg_z - peg_z), z_reward_coef)
                else:
                    # curr should > prev
                    z_reward = stable_mul((peg_z - self.prev_peg_z), z_reward_coef)
                self.prev_peg_z = peg_z

            reward = (
                obj_to_target_reward
                + gripper_reward
                + tcp_to_obj_reward
                + z_reward
                + terminal_reward
                + irr_penalty
            )
            return (
                reward,
                dict(
                    step_reward=reward,
                    success=success,
                    tcp_to_obj=tcp_to_obj,
                    tcp_to_obj_reward=tcp_to_obj_reward,
                    gripper_reward=gripper_reward,
                    obj_to_target=obj_to_target,
                    obj_to_target_reward=obj_to_target_reward,
                    z_reward=z_reward,
                    irrevresible=self.is_irreversible,
                ),
            )
        else:
            raise NotImplementedError

    def is_irreversible_state(self) -> bool:
        """deprecated as irreversible maybe not explicit."""
        # obj_head = self._get_site_pos("pegHead")
        # obj_end = self._get_site_pos("pegEnd")
        # return (not self.reversible_space.contains(obj_head)) or (
        #     not self.reversible_space.contains(obj_end)
        # )
        if self.reset_if_explicit_irreversible:
            return self._get_site_pos("pegHead")[2] < -0.05
        return False

    def is_successful(self, obs: Optional[np.ndarray] = None) -> bool:
        if obs is None:
            obs = self._get_flatten_full_obs()
        return np.linalg.norm(obs[4:7] - self._target_pos) <= self.TARGET_RADIUS
