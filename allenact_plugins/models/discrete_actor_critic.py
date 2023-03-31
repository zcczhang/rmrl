from collections import OrderedDict
from typing import Dict, Union, Optional, Tuple, Any, Sequence, cast, Callable

import gym
import numpy as np
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, DistributionType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.utils.model_utils import debug_model_info
from allenact_plugins.distributions.distributions import (
    MultiCategorical,
    DictActionSpaceDistr,
    GaussianDistr,
)
from allenact_plugins.models.nn.mlp import MLPActorCritic


class DiscreteActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: Union[gym.spaces.MultiDiscrete, gym.spaces.Discrete],
        observation_space: gym.spaces.Dict,
        num_stacked_frames: Optional[int] = 4,
        # for actor-critic MLP
        mlp_hidden_dims: Sequence[int] = (512, 256),
        activation_fn: Callable = nn.Tanh,
        shared_state_encoder: bool = True,
        separate_state_encoder: bool = False,
        critic_condition_action: bool = False,
        use_action_embed: bool = True,
        action_embed_dim: Optional[int] = 32,
        obs_feature_dim: int = 512,
        squash_output: bool = True,
        orthogonal_init: bool = False,
        # for Gaussian in dict action space
        use_log_std: bool = False,
        min_action_std: float = 0.1,
        max_action_std: float = 0.5,
    ):
        super().__init__(action_space, observation_space)
        self.input_uuid = input_uuid
        state_dim = np.product(observation_space[self.input_uuid].shape)
        if isinstance(action_space, gym.spaces.Dict):
            dict_space_dims = OrderedDict()
            for key in action_space:
                space = action_space[key]
                if isinstance(space, gym.spaces.Discrete):
                    dict_space_dims["discrete"] = (key, space.n)
                elif isinstance(space, gym.spaces.Box):
                    dict_space_dims["box"] = (key, space.shape[0])
                else:
                    raise NotImplementedError(space)
            action_dim = sum([v[1] for v in dict_space_dims.values()])
            self.dict_space_dims = dict_space_dims
            self.is_multi_discrete = False
            self.is_dict_space = True
            self.use_log_std = use_log_std
            self.min_action_std = min_action_std
            self.max_action_std = max_action_std
            std_init = min_action_std if self.use_log_std else 0.0
            self.std = nn.Parameter(
                torch.ones(dict_space_dims["box"][1], dtype=torch.float32) * std_init,
                requires_grad=True,
            )
        else:
            action_dim = action_space.shape
            self.is_multi_discrete = len(action_dim) != 0
            action_dim = (
                np.sum(action_space.nvec) if self.is_multi_discrete else action_space.n
            )
            self.is_dict_space = False

        self.actor_critic = MLPActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            env_action_dim=len(action_space.nvec)
            if self.is_multi_discrete
            else action_dim,  # careful with dict action space
            hidden_dim=mlp_hidden_dims,
            num_stacked_frames=num_stacked_frames,
            activation_fn=activation_fn,
            shared_state_encoder=shared_state_encoder,
            separate_state_encoder=separate_state_encoder,
            critic_condition_action=critic_condition_action,
            use_action_embed=use_action_embed,
            action_embed_dim=action_embed_dim,
            obs_feature_dim=obs_feature_dim,
            squash_output=squash_output,
            orthogonal_init=orthogonal_init,
        )
        debug_model_info(self)

    def _recurrent_memory_specification(self):
        return None

    def forward(
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        obs = observations[self.input_uuid]
        means, values, _ = self.actor_critic(obs, prev_actions=prev_actions)

        if self.is_multi_discrete:  # preferred
            distr = MultiCategorical(logits=means, action_dim=self.action_space.nvec.tolist())  # type: ignore
        elif self.is_dict_space:
            ordered_keys = [v[0] for v in self.dict_space_dims.values()]
            is_discrete_first = ordered_keys[0] == "discrete"
            disc_key = self.dict_space_dims["discrete"][0]
            cont_key = self.dict_space_dims["box"][0]
            disc_dim = self.dict_space_dims["discrete"][1]
            cont_dim = self.dict_space_dims["box"][1]
            disc_means = (
                means[..., :disc_dim] if is_discrete_first else means[..., cont_dim:]
            )
            cont_means = (
                means[..., disc_dim:] if is_discrete_first else means[..., :cont_dim]
            )
            std = (
                self.std
                if self.use_log_std
                else (
                    self.min_action_std
                    + (self.max_action_std - self.min_action_std)
                    * torch.sigmoid(self.std)
                )
            )
            if self.use_log_std:
                std = torch.exp(std)
            distr = DictActionSpaceDistr(
                cont_distr=(cont_key, GaussianDistr(loc=cont_means, scale=std)),
                disc_distr=(disc_key, CategoricalDistr(logits=disc_means)),
                ordered_keys=ordered_keys,
            )
        else:
            distr = CategoricalDistr(logits=means)
        return (
            ActorCriticOutput(cast(DistributionType, distr), values, {},),
            None,  # no Memory
        )
