from typing import (
    Dict,
    Union,
    Optional,
    Tuple,
    Any,
    Sequence,
    cast,
    List,
    Callable,
)

import gym
import numpy as np
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.policy import ActorCriticModel, DistributionType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.utils.model_utils import debug_model_info
from allenact.utils.system import get_logger
from allenact_plugins.distributions.distributions import MultiCategorical
from allenact_plugins.models.nn.cnn import SimpleCNN
from allenact_plugins.models.nn.mlp import MLPActorCritic
from allenact_plugins.models.nn.resnet import ResNetEncoder


class DiscreteActorCriticVisual(ActorCriticModel[CategoricalDistr]):
    """Actor Critic model for multi-discrete action space with visual
    observation.

    Args:
        input_uuid: uuid for observation sensor
        action_space: action space (multi-discrete)
        observation_space: env observation space
        num_stacked_frames: number of frame stacking (channel-wise)
        mlp_hidden_dims: mlp hidden dimension for actor and critic
        rgb_uuid: uuid (key) for rgb image observation, None for not including
        state_uuid: uuid (key) for low-dim observation, None for not including
        visual_encoder: name for visual obs encoder if having rgb obs
        channel_first: whether rgb obs channel first
        visual_output_size: output dim for fc layer followed by visual encoder
        low_dim_feature_dim: output dim for 1-layer encoder for low-dim obs, None for no encoder
        visual_encoder_kwargs: kwargs for initializing the visual encoder
    """

    def __init__(
        self,
        action_space: Union[gym.spaces.MultiDiscrete, gym.spaces.Discrete],
        observation_space: gym.spaces.Dict,
        input_uuid: Optional[str] = None,
        num_stacked_frames: Optional[int] = None,
        # for actor-critic MLP
        mlp_hidden_dims: Sequence[int] = (512, 256),
        low_dim_feature_dim: Optional[int] = 256,
        activation_fn: Callable = nn.Tanh,
        shared_state_encoder: bool = True,
        separate_state_encoder: bool = False,
        critic_condition_action: bool = False,
        use_action_embed: bool = False,
        action_embed_dim: Optional[int] = None,
        squash_output: bool = True,
        orthogonal_init: bool = False,
        # visual encoder(s)
        rgb_uuid: Optional[Union[str, List[str]]] = "rgb",
        state_uuid: Optional[str] = "state",
        visual_encoder: Optional[Union[Tuple[str, ...], str]] = ("cnn",),
        single_visual_encoder: bool = True,
        channel_first: bool = True,
        visual_output_size: Optional[int] = 512,
        visual_encoder_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if kwargs is not None and len(kwargs) != 0:
            get_logger().warning(f"unused model kwargs {kwargs}")
        super().__init__(action_space, observation_space)
        action_dim = action_space.shape
        self.is_multi_discrete = len(action_dim) != 0
        action_dim = (
            np.sum(action_space.nvec) if self.is_multi_discrete else action_space.n
        )
        self.input_uuid = input_uuid
        if input_uuid is not None and isinstance(
            observation_space[input_uuid], gym.spaces.Dict
        ):
            self._dict_obs = True
            space = observation_space[input_uuid]
        else:
            self._dict_obs = False
            space = observation_space
        state_dim = np.product(space[state_uuid].shape)
        env_action_dim = (
            len(action_space.nvec) if self.is_multi_discrete else action_dim
        )

        if isinstance(rgb_uuid, str):
            rgb_uuid = [rgb_uuid]
        self.rgb_uuid = list(rgb_uuid) if rgb_uuid is not None else []
        self.state_uuid = state_uuid

        visual_features = (visual_output_size or 0) * len(self.rgb_uuid)
        self._has_state_obs = (
            state_uuid in space.spaces and low_dim_feature_dim is not None
        )
        if visual_features > 0:
            encoder_kwargs = dict(
                observation_space=space,
                output_size=visual_output_size,
                num_stacked_frames=num_stacked_frames or 1,
                channel_first=channel_first,
            )
            encoder_kwargs.update(visual_encoder_kwargs or {})
            self.single_visual_encoder = single_visual_encoder
            if not single_visual_encoder and len(self.rgb_uuid) > 1:
                # one encoder for one view
                self.vis_encoder = []
                if len(self.rgb_uuid) > 1 and isinstance(visual_encoder, str):
                    visual_encoder = [visual_encoder] * len(self.rgb_uuid)
                assert len(self.rgb_uuid) == len(visual_encoder)
                for view, encoder in zip(self.rgb_uuid, visual_encoder):
                    encoder_kwargs.update(rgb_uuid=view,)
                    self.vis_encoder.append(VISUAL_BACKBONES[encoder](**encoder_kwargs))
                assert len(self.vis_encoder) in [
                    1,
                    2,
                ], f"only support at most two views"
                self.vis_encoder1, self.vis_encoder2 = self.vis_encoder

            else:
                # single encoder for every rgb obs
                if not isinstance(visual_encoder, str):
                    assert len(visual_encoder) == 1, visual_encoder
                    visual_encoder = visual_encoder[0]
                encoder_kwargs.update(rgb_uuid=self.rgb_uuid[0],)
                self.vis_encoder = VISUAL_BACKBONES[visual_encoder](**encoder_kwargs)

        self.actor_critic = MLPActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,  # dof * num_bins
            env_action_dim=env_action_dim,  # dof
            hidden_dim=mlp_hidden_dims,
            extra_feature_dim=visual_features,
            num_stacked_frames=num_stacked_frames,
            activation_fn=activation_fn,
            shared_state_encoder=shared_state_encoder,
            separate_state_encoder=separate_state_encoder,
            critic_condition_action=critic_condition_action,
            use_action_embed=use_action_embed,
            action_embed_dim=action_embed_dim,
            obs_feature_dim=low_dim_feature_dim,
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
        nsteps, nsamplers, nagents = masks.shape[:3]
        obs = observations[self.input_uuid] if self._dict_obs else observations
        visual_rep = []
        for i, view in enumerate(self.rgb_uuid):
            rgb = obs[view]
            if not self.single_visual_encoder:
                rgb_rep = self.vis_encoder[i](rgb)
            else:
                rgb_rep = self.vis_encoder(rgb)
            rgb_rep = rgb_rep.view(nsteps, nsamplers * nagents, -1)
            visual_rep.append(rgb_rep)

        means, values, _ = self.actor_critic(
            obs[self.state_uuid].view(nsteps, nsamplers * nagents, -1).float(),
            extra_rep=torch.cat(visual_rep, dim=-1),
            prev_actions=prev_actions,
        )
        if self.is_multi_discrete:
            distr = MultiCategorical(logits=means, action_dim=self.action_space.nvec.tolist())  # type: ignore
        else:
            distr = CategoricalDistr(logits=means)
        return (
            ActorCriticOutput(cast(DistributionType, distr), values, {}),
            None,  # no Memory
        )


VISUAL_BACKBONES = dict(cnn=SimpleCNN, resnet=ResNetEncoder)
