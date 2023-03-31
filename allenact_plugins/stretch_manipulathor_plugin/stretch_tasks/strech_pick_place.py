from allenact.base_abstractions.misc import RLStepResult
from allenact_plugins.stretch_manipulathor_plugin.stretch_constants import (
    STRETCH_ACTIONS,
    PICKUP,
    STRETCH_ACTIONS_FULL,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_tasks.stretch_task_base import (
    AbstractStretchTask,
)
from allenact_plugins.stretch_manipulathor_plugin.stretch_utils import close_enough


__all__ = ["StretchPickPlace", "StretchPickPlaceFullAction"]


class StretchPickPlace(AbstractStretchTask):

    _actions = STRETCH_ACTIONS

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]
        self._last_action_str = action_str
        action_dict = {"action": action_str}
        object_id = self.source_obj_id
        if action_str == PICKUP:
            action_dict = {**action_dict, "object_id": object_id}
        self.env.step(action_dict)
        self._last_action_success = self.env.last_action_success

        last_action_name = self._last_action_str
        last_action_success = float(self.last_action_success)
        self.action_sequence_and_success.append((last_action_name, last_action_success))

        first_grasped = False
        if not self.object_first_picked_up:
            if self.env.is_object_at_low_level_hand(object_id):
                self.object_first_picked_up = True
                self.eplen_pickup = (
                    self._num_steps_taken + 1
                )  # plus one because this step has not been counted yet
                first_grasped = True

        if self.success_criteria(threshold=self.success_threshold):
            self._took_end_action = True
            self._last_action_success = True
            self._success = True
        elif self.terminate_if_obj_dropped and self.is_obj_off_table():
            self._took_end_action = True
            self._last_action_success = False
            self._success = False
            self.debug_info["obj_off_table"] = 1.0
        elif self.terminate_if_obj_dropped:
            self.debug_info["obj_off_table"] = 0.0

        reward = float(self.judge(first_grasped))
        self._rewards.append(reward)
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    def success_criteria(self, threshold: float = 0.1) -> bool:
        """source obj is picked and placed within the goal obj."""
        # if not self.object_first_picked_up:   # not necessary for reset free
        #     return False
        source_state = self.source_obj_state
        if not self.is_obj_goal:
            goal_achieved = close_enough(
                source_state, self.goal_pos, threshold=threshold, norm=2
            )
        else:
            # 1. x & z axis has to be close enough
            goal_state = self.obj_goal_state
            goal_achieved = close_enough(
                source_state,
                self.goal_pos,
                threshold=threshold,
                norm=1,
                axis=("x", "z"),
            )

            if self.add_debug_metrics:
                for i in ["x", "y", "z"]:
                    self.debug_info[f"obj2goal_{i}_dist"] = abs(
                        source_state["position"][i] - goal_state["position"][i]
                    )
                self.debug_info["obj2goal_close_enough"] = int(goal_achieved)

            # 2. receptacle trigger
            if "receptacleObjectIds" in goal_state.keys():
                obj_contact_receptacle = (
                    source_state["objectId"] in goal_state["receptacleObjectIds"]
                )
                if self.add_debug_metrics:
                    self.debug_info["obj_contact_receptacle"] = int(
                        obj_contact_receptacle
                    )

                goal_achieved &= obj_contact_receptacle

            else:
                # change if doing cube stacking tasks
                assert goal_state["receptacle"], f"{self.goal_obj_id} not receptacle"
        return goal_achieved

    def judge(self, first_grasped: bool) -> float:
        action_penalty = self.reward_configs["step_penalty"]
        if not self.last_action_success or (
            self._last_action_str == PICKUP and not self.object_first_picked_up
        ):
            action_penalty += self.reward_configs["failed_action_penalty"]

        hand_reward = 0
        if first_grasped:
            hand_reward += self.reward_configs["first_picked_up_reward"]

        current_arm_to_obj_distance = self.arm_distance_from_obj()
        if self.last_arm_to_obj_distance is None:
            self.last_arm_to_obj_distance = current_arm_to_obj_distance
        hand_reward += (
            self.last_arm_to_obj_distance - current_arm_to_obj_distance
        ) * self.reward_configs["arm_dist_multiplier"]
        self.last_arm_to_obj_distance = current_arm_to_obj_distance

        obj_to_target_reward = 0
        current_obj_to_goal_distance = self.obj_distance_from_goal()
        if self.last_obj_to_goal_distance is None:
            self.last_obj_to_goal_distance = current_obj_to_goal_distance
        obj_to_target_reward += (
            self.last_obj_to_goal_distance - current_obj_to_goal_distance
        ) * self.reward_configs["obj_dist_multiplier"]
        self.last_obj_to_goal_distance = current_obj_to_goal_distance

        terminal_reward = 0
        if self._success:
            assert self._took_end_action
            terminal_reward += self.reward_configs["goal_success_reward"]

        step_reward = (
            action_penalty + hand_reward + obj_to_target_reward + terminal_reward
        )
        self.reward_info = dict(
            step_reward=step_reward,
            action_penalty=action_penalty,
            hand_reward=hand_reward,
            obj_to_target_reward=obj_to_target_reward,
            terminal_reward=terminal_reward,
        )

        return step_reward


class StretchPickPlaceFullAction(StretchPickPlace):
    _actions = STRETCH_ACTIONS_FULL
