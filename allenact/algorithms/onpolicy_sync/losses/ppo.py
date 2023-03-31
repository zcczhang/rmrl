"""Defining the PPO loss for actor critic type models."""

from typing import Dict, Optional, Callable, cast, Tuple

import numpy as np
import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class PPO(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization loss.

    # Attributes

    clip_param : The clipping parameter to use.
    value_loss_coef : Weight of the value loss.
    entropy_coef : Weight of the entropy (encouraging) loss.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    clip_decay : Callable for clip param decay factor (function of the current number of steps)
    entropy_method_name : Name of Distr's entropy method name. Default is `entropy`,
                          but we might use `conditional_entropy` for `SequentialDistr`
    show_ratios : If True, adds tracking for the PPO ratio (linear, clamped, and used) in each
                  epoch to be logged by the engine.
    normalize_advantage: Whether or not to use normalized advantage. Default is True.
    """

    def __init__(
        self,
        clip_param: float,
        value_loss_coef: float,
        entropy_coef: float,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        entropy_method_name: str = "entropy",
        normalize_advantage: bool = True,
        show_ratios: bool = False,
        *args,
        **kwargs,
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        # clip_param_end = kwargs.pop("clip_param_end", clip_param)
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        # self.clip_decay = (
        #     clip_decay
        #     if clip_decay is not None
        #     else (
        #         lambda x: 1 - (1 - clip_param_end/clip_param) * (max(1, x) / max_steps)
        #     )
        # )
        self.entropy_method_name = entropy_method_name
        self.show_ratios = show_ratios
        if normalize_advantage:
            self.adv_key = "norm_adv_targ"
        else:
            self.adv_key = "adv_targ"

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:  # TODO tuple output

        actions = cast(torch.LongTensor, batch["actions"])
        values = actor_critic_output.values

        action_log_probs = actor_critic_output.distributions.log_prob(actions)
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(batch[self.adv_key].shape)
            return t.view(
                t.shape + ((1,) * (len(batch[self.adv_key].shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * batch[self.adv_key]
        surr2 = clamped_ratio * batch[self.adv_key]

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        if self.use_clipped_value_loss:
            value_pred_clipped = batch["values"] + (values - batch["values"]).clamp(
                -clip_param, clip_param
            )
            value_losses = (values - batch["returns"]).pow(2)
            value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
        else:
            value_loss = 0.5 * (cast(torch.FloatTensor, batch["returns"]) - values).pow(
                2
            )

        step_loss_out = {
            "value": (value_loss, self.value_loss_coef),
            "action": (action_loss, None),
            "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
        }
        extras = actor_critic_output.extras
        if "std" in extras.keys():
            step_loss_out["std"] = (extras["std"], 0)

        if "irr_distributions" in extras.keys():
            # binary cross entropy loss for irreversible losses
            weighted_BCE_loss = 0
            if step_count > 1:
                irr_distributions = extras["irr_distributions"]
                irr_sensors = (
                    batch["observations"]["irr_sensor"].squeeze(-1).to(torch.float32)
                )

                num_irr = int(torch.sum(irr_sensors))  # num of irreversible
                num_r = np.product(irr_sensors.shape) - num_irr  # num of reversible
                w_irr = -(1 - num_irr) / np.product(irr_sensors.shape)
                w_r = -(1 - num_r) / np.product(irr_sensors.shape)
                weight = torch.where(irr_sensors == 1, w_irr, w_r)

                def _positive(x: torch.Tensor) -> torch.Tensor:
                    """prevent exp(log_prob) > 1 for normal distribution.

                    https://github.com/pytorch/pytorch/issues/7637#issuecomment-389963143
                    """
                    return x.relu() + 1e-7

                weighted_BCE_loss = -(
                    (
                        (
                            irr_sensors.detach() * irr_distributions.log_prob(actions)
                            + (1 - irr_sensors.detach())
                            * torch.log(
                                _positive(
                                    1 - torch.exp(irr_distributions.log_prob(actions))
                                )
                            )
                        )
                        * weight.reshape(irr_sensors.shape)
                    ).sum(-1)
                ).mean()
            step_loss_out["irreversible BCE"] = (weighted_BCE_loss, None)

        return (
            step_loss_out,
            {
                "ratio": ratio,
                "ratio_clamped": clamped_ratio,
                "ratio_used": torch.where(
                    cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                ),
            }
            if self.show_ratios
            else {},
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            },
            {key: float(value.mean().item()) for key, value in ratio_info.items()},
        )

        return result if self.show_ratios else result[:2]


class RNDPPO(AbstractActorCriticLoss):
    """Implementation of the Proximal Policy Optimization with Random Network
    Distillation loss.

    # Attributes

    clip_param : The clipping parameter to use.
    value_loss_coef : Weight of the value loss.
    entropy_coef : Weight of the entropy (encouraging) loss.
    use_clipped_value_loss : Whether or not to also clip the value loss.
    clip_decay : Callable for clip param decay factor (function of the current number of steps)
    entropy_method_name : Name of Distr's entropy method name. Default is `entropy`,
                          but we might use `conditional_entropy` for `SequentialDistr`
    show_ratios : If True, adds tracking for the PPO ratio (linear, clamped, and used) in each
                  epoch to be logged by the engine.
    normalize_advantage: Whether or not to use normalized advantage. Default is True.
    """

    def __init__(
        self,
        clip_param: float,
        value_loss_coef: float,
        entropy_coef: float,
        use_clipped_value_loss=True,
        clip_decay: Optional[Callable[[int], float]] = None,
        entropy_method_name: str = "entropy",
        normalize_advantage: bool = True,
        show_ratios: bool = False,
        *args,
        **kwargs,
    ):
        """Initializer.

        See the class documentation for parameter definitions.
        """
        super().__init__(*args, **kwargs)
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_clipped_value_loss = use_clipped_value_loss
        self.clip_decay = clip_decay if clip_decay is not None else (lambda x: 1.0)
        self.entropy_method_name = entropy_method_name
        self.show_ratios = show_ratios
        if normalize_advantage:
            self.adv_key = "norm_adv_targ"
        else:
            self.adv_key = "adv_targ"

        self.batch_int_value = None
        self.init_step_count = None

    def loss_per_step(
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, Optional[float]]], Dict[str, torch.Tensor]
    ]:
        """in order to keep the `next_value`, the 'real' last values will be
        discarded,

        i.e. step now should be step - 1. For now, only `mini_batch_size=1` supported
        """
        # (step, b, 4), the log_prob will be (step - 1, b, 4)
        actions = cast(torch.LongTensor, batch["actions"])
        # (step - 1, b, 1)
        ext_values = actor_critic_output.values[:-1]
        # (step, b, 1)
        int_values = actor_critic_output.extras["values_internal"]
        # (step - 1, b, 1)
        int_reward = actor_critic_output.extras["intrinsic_reward"][:-1].detach()
        # (step, b, 1)
        masks = batch["masks"]
        # (step, b, 1)
        ext_returns = batch["returns"]

        def compute_returns_and_advantages(
            use_gae: bool = True, gamma: float = 0.99, tau: float = 0.95,
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
            if self.batch_int_value is None:
                # batch-wise "previous" values
                # (step, b, 1)
                self.batch_int_value = int_values.detach()
            # (step, b, 1)
            int_returns = torch.zeros_like(
                ext_returns, dtype=ext_returns.dtype, device=ext_returns.device
            )

            if use_gae:
                gae = 0
                for step in reversed(range(int_reward.shape[0])):
                    delta = (
                        int_reward[step]
                        + gamma * self.batch_int_value[step + 1] * masks[step + 1]
                        - self.batch_int_value[step]
                    )
                    gae = delta + gamma * tau * masks[step + 1] * gae
                    int_returns[step] = gae + self.batch_int_value[step]
            else:
                raise NotImplementedError("only support using gae!")

            # calculate internal advantages
            int_advantages = int_returns[:-1] - self.batch_int_value[:-1]
            int_advantages = (
                int_advantages
                if self.adv_key != "norm_adv_targ"
                else (
                    (int_advantages - int_advantages.mean())
                    / (int_advantages.std() + 1e-5)
                )
            )
            return int_returns[:-1], int_advantages

        # get internal returns
        int_returns, int_advantages = compute_returns_and_advantages(
            **actor_critic_output.extras["int_r_kwargs"]
        )
        # (step - 1, b, 1)
        ext_advantages = batch[self.adv_key][:-1]
        # in the paper, R = R_E + R_I
        comb_advantages = ext_advantages + int_advantages

        # (step - 1, b, 1)
        action_log_probs = actor_critic_output.distributions.log_prob(actions)[:-1]
        dist_entropy: torch.FloatTensor = getattr(
            actor_critic_output.distributions, self.entropy_method_name
        )()

        def add_trailing_dims(t: torch.Tensor):
            assert len(t.shape) <= len(comb_advantages.shape)
            return t.view(
                t.shape + ((1,) * (len(comb_advantages.shape) - len(t.shape)))
            )

        dist_entropy = add_trailing_dims(dist_entropy)

        clip_param = self.clip_param * self.clip_decay(step_count)

        ratio = torch.exp(action_log_probs - batch["old_action_log_probs"][:-1])
        ratio = add_trailing_dims(ratio)
        clamped_ratio = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)

        surr1 = ratio * comb_advantages
        surr2 = clamped_ratio * comb_advantages

        use_clamped = surr2 < surr1
        action_loss = -torch.where(cast(torch.Tensor, use_clamped), surr2, surr1)

        def _get_value_loss(
            batch_values: torch.Tensor, values: torch.Tensor, returns: torch.Tensor
        ) -> torch.Tensor:
            assert batch_values.shape[0] == values.shape[0] == returns.shape[0]
            if self.use_clipped_value_loss:
                value_pred_clipped = batch_values + (values - batch_values).clamp(
                    -clip_param, clip_param
                )
                value_losses = (values - returns).pow(2)
                value_losses_clipped = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
            else:
                value_loss = 0.5 * (cast(torch.FloatTensor, returns) - values).pow(2)
            return value_loss

        ext_loss = _get_value_loss(
            batch_values=batch["values"][:-1],  # (step - 1, b, 1)
            values=ext_values,  # (step - 1, b, 1)
            returns=ext_returns[:-1],  # (step - 1, b, 1)
        )
        int_loss = _get_value_loss(
            batch_values=self.batch_int_value[:-1],  # (step - 1, b, 1)
            values=int_values[:-1],  # (step - 1, b, 1)
            returns=int_returns,  # (step - 1, b, 1)
        )
        comb_value_loss = ext_loss + int_loss  # (step - 1, b, 1)

        # (step - 1, b, 1) forward loss
        int_mse_loss = actor_critic_output.extras["intrinsic_reward"][:-1]

        # (step, b, 1)
        self.batch_int_value = int_values.detach()

        # noinspection PyUnresolvedReferences
        return (
            {
                "value": (comb_value_loss, self.value_loss_coef),
                "action": (action_loss, None),
                "entropy": (dist_entropy.mul_(-1.0), self.entropy_coef),  # type: ignore
                "forward_mse_loss": (int_mse_loss, None),  # weight 1
                "intrinsic_loss": (int_loss, 0),  # log the intrinsic loss
                "extrinsic_loss": (ext_loss, 0),  # log the extrinsic loss
            },
            {
                "ratio": ratio,
                "ratio_clamped": clamped_ratio,
                "ratio_used": torch.where(
                    cast(torch.Tensor, use_clamped), clamped_ratio, ratio
                ),
            }
            if self.show_ratios
            else {},
        )

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs,
    ):
        # update batch_int_value
        if self.init_step_count is None:
            self.init_step_count = step_count
        elif self.init_step_count != step_count:
            self.batch_int_value = None
            self.init_step_count = step_count

        losses_per_step, ratio_info = self.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses = {
            key: (loss.mean(), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        result = (
            total_loss,
            {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            },
            {key: float(value.mean().item()) for key, value in ratio_info.items()},
        )

        return result if self.show_ratios else result[:2]


PPOConfig = dict(clip_param=0.1, value_loss_coef=0.5, entropy_coef=0.01)
