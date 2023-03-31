from typing import (
    Dict,
    Union,
    Optional,
    Tuple,
    Any,
    List,
    Sequence,
    Callable,
)

import gym
import numpy as np
import torch
from torch import nn

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.utils.model_utils import debug_model_info, FeatureEmbedding
from allenact.utils.system import get_logger
from allenact_plugins.distributions.distributions import MultiCategorical
from allenact_plugins.models.nn.cnn import SimpleCNN
from allenact_plugins.models.nn.mlp import one_layer_state_encoder, make_mlp_hidden
from allenact_plugins.models.nn.resnet import ResNetEncoder
from allenact_plugins.navigation_plugin.objectnav.models import ResnetTensorGoalEncoder


class DiscreteActorCriticVisualWithPrompt(ActorCriticModel[CategoricalDistr]):
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
        mlp_hidden_dims: Optional[Sequence[int]] = (),
        mlp_activation_fn: Callable = nn.ReLU,
        ortho_init: bool = False,
        low_dim_feature_dim: Optional[int] = 512,
        goal_embed_dims: Optional[int] = 512,
        shared_state_encoder: bool = True,
        condition_on_prev_action: bool = False,
        action_embed_dim: Optional[int] = None,
        # sensor/modalities uuid
        prompt_img_uuid: Optional[str] = None,
        rgb_uuid: Optional[Union[str, List[str]]] = None,
        state_uuid: Optional[str] = None,
        prompt_uuid: Optional[str] = None,
        goal_uuid: Optional[str] = None,
        # visual encoder(s)
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        # Optional pickup embedding sensor
        obj_in_hand_uuid: Optional[str] = None,
        obj_in_hand_embedding_dim: Optional[int] = 32,
        extra_discrete_sensor_uuid: Optional[str] = None,
        extra_discrete_sensor_dim: Optional[int] = None,
        extra_discrete_sensor_embedding_dim: Optional[int] = None,
        # rnn kwargs, not use
        use_rnn=False,
        rnn_hidden_dim: int = 512,
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers: int = 1,
        rnn_type: str = "GRU",
        **kwargs,
    ):
        if kwargs is not None and len(kwargs) != 0:
            get_logger().warning(f"{self.__class__.__name__} unused kwargs {kwargs}")
        self.action_space = action_space
        super().__init__(action_space, observation_space)
        action_dim = action_space.shape
        self.is_multi_discrete = len(action_dim) != 0
        action_dim = (
            np.sum(action_space.nvec) if self.is_multi_discrete else action_space.n
        )
        self.action_dim = action_dim

        self.prompt_img_uuid = prompt_img_uuid
        self.prompt_uuid = prompt_uuid
        self.rgb_uuid = rgb_uuid
        self.state_uuid = state_uuid
        self.goal_uuid = goal_uuid
        assert (
            goal_uuid is None or prompt_uuid is None
        ), f"goal sensor or prompt sensor should only be one"
        self.prompt_goal = prompt_uuid is not None
        self.goal_embed_dims = goal_embed_dims

        rgb_space = None
        if state_uuid is not None and state_uuid in observation_space.spaces:
            state_space = observation_space[state_uuid]
        elif (
            input_uuid is not None
            and input_uuid in observation_space.spaces
            and isinstance(observation_space[input_uuid], gym.spaces.Dict)
        ):
            state_space = observation_space[input_uuid][state_uuid]
            rgb_space = observation_space[input_uuid][rgb_uuid]
        else:
            state_space = None
        state_dim = np.product(state_space.shape) if state_space is not None else 0

        if self.prompt_goal:
            if prompt_img_uuid in observation_space.spaces:
                space = observation_space[prompt_img_uuid]
                assert prompt_uuid in space.spaces, space
                assert rgb_uuid in space.spaces, space
                prompt_space = space[prompt_uuid]
                rgb_space = space[rgb_uuid]
            else:
                raise NotImplementedError(observation_space)
            task_goal_uuid = prompt_uuid
            task_goal_space = prompt_space  # consider prompt as goal
        elif goal_uuid is not None and goal_uuid in observation_space.spaces:
            task_goal_uuid = goal_uuid
            task_goal_space = observation_space[goal_uuid]
        else:
            raise NotImplementedError(observation_space)

        if rgb_space is None:
            assert rgb_uuid in observation_space.spaces
            rgb_space = observation_space[rgb_uuid]

        self.goal_visual_encoder = ResnetTensorGoalEncoder(
            observation_spaces=gym.spaces.Dict(
                {rgb_uuid: rgb_space, task_goal_uuid: task_goal_space}
            ),
            goal_sensor_uuid=task_goal_uuid,
            resnet_preprocessor_uuid=rgb_uuid,
            goal_embed_dims=goal_embed_dims,
            resnet_compressor_hidden_out_dims=resnet_compressor_hidden_out_dims,
            combiner_hidden_out_dims=combiner_hidden_out_dims,
        )

        self.state_encoder = one_layer_state_encoder(
            state_dim=state_dim,
            output_size=low_dim_feature_dim if shared_state_encoder else None,
            num_stacked_frames=num_stacked_frames or 1,
        )
        ac_input_dim = self.goal_visual_encoder.output_dims + low_dim_feature_dim * (
            state_dim not in [None, 0]
        )

        add_prev_action_null_token: bool = False
        action_embed_dim = action_embed_dim if condition_on_prev_action else 0
        assert action_embed_dim is not None, action_embed_dim
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_dim,
            output_size=action_embed_dim,
        )
        ac_input_dim += action_embed_dim

        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn = RNNStateEncoder(
                input_size=ac_input_dim,
                hidden_size=rnn_hidden_dim,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
            )
            self.rnn_hidden_dim = rnn_hidden_dim
            ac_input_dim = rnn_hidden_dim

        self.obj_in_hand_uuid = obj_in_hand_uuid
        if obj_in_hand_uuid is not None:
            assert obj_in_hand_embedding_dim is not None
            self.obj_in_hand_embedding = nn.Embedding(
                num_embeddings=2, embedding_dim=obj_in_hand_embedding_dim
            )
            ac_input_dim += obj_in_hand_embedding_dim

        self.extra_discrete_sensor_uuid = extra_discrete_sensor_uuid
        if extra_discrete_sensor_uuid is not None:
            assert (
                extra_discrete_sensor_dim is not None
                and extra_discrete_sensor_embedding_dim is not None
            )
            self.extra_discrete_sensor_embedding = nn.Embedding(
                num_embeddings=extra_discrete_sensor_dim,
                embedding_dim=extra_discrete_sensor_embedding_dim,
            )
            ac_input_dim += extra_discrete_sensor_embedding_dim

        self.actor, self.critic = self.make_actor_critic(
            ac_input_dim=ac_input_dim,
            hidden_dim=mlp_hidden_dims,
            activation_fn=mlp_activation_fn,
            ortho_init=ortho_init,
        )

        debug_model_info(self)

    def make_actor_critic(
        self,
        ac_input_dim: int,
        hidden_dim: Optional[Sequence[int]],
        activation_fn: Callable = nn.ReLU,
        ortho_init: bool = False,
    ) -> tuple:
        if isinstance(hidden_dim, int):
            hidden_dim = (hidden_dim,)
        if hidden_dim is None:
            hidden_dim = ()
        hidden_dim = (ac_input_dim,) + tuple(hidden_dim)
        actor = nn.Sequential(
            *make_mlp_hidden(activation_fn, *hidden_dim),
            nn.Linear(hidden_dim[-1], self.action_dim),
        )
        if ortho_init:
            for module in actor.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=0.01)
                    nn.init.constant_(module.bias, 0)
        critic = nn.Sequential(
            *make_mlp_hidden(activation_fn, *hidden_dim), nn.Linear(hidden_dim[-1], 1),
        )
        if ortho_init:
            for module in critic.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight)
                    nn.init.constant_(module.bias, 0)
        return actor, critic

    def _recurrent_memory_specification(self) -> Optional[dict]:
        return (
            dict(
                rnn=(
                    (
                        ("layer", self.num_recurrent_layers),
                        ("sampler", None),
                        ("hidden", self.recurrent_hidden_state_size),
                    ),
                    torch.float32,
                )
            )
            if self.use_rnn
            else None
        )

    @property
    def recurrent_hidden_state_size(self) -> Optional[int]:
        """The recurrent hidden state size of the model, None for MLP."""
        return self.rnn_hidden_dim if self.use_rnn else None

    @property
    def num_recurrent_layers(self) -> Optional[int]:
        """Number of recurrent hidden layers, None for MLP."""
        return self.state_encoder.num_recurrent_layers if self.use_rnn else None

    def forward(
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        nsteps, nsamplers, nagents = masks.shape[:3]
        rgb_prompt_obs = (
            observations[self.prompt_img_uuid] if self.prompt_goal else observations
        )
        visual_goal_embeds = self.goal_visual_encoder(rgb_prompt_obs)
        if self.state_uuid is not None:
            low_dim_state_embeds = self.state_encoder(observations[self.state_uuid])
            joint_embeds = [visual_goal_embeds, low_dim_state_embeds]
        else:
            joint_embeds = [visual_goal_embeds]

        if self.prev_action_embedder.input_size == self.action_dim + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds.append(prev_actions_embeds)

        if self.obj_in_hand_uuid is not None:
            in_hand = observations[self.obj_in_hand_uuid].long()
            in_hand_embed = self.obj_in_hand_embedding(in_hand).view(
                nsteps, nsamplers * nagents, -1
            )
            joint_embeds.append(in_hand_embed)

        if self.extra_discrete_sensor_uuid is not None:
            extra_obs = observations[self.extra_discrete_sensor_uuid].long()
            extra_obs_embed = self.extra_discrete_sensor_embedding(extra_obs).view(
                nsteps, nsamplers * nagents, -1
            )
            joint_embeds.append(extra_obs_embed)

        rep = torch.cat(joint_embeds, dim=-1)

        rnn_hidden_states = None
        if self.use_rnn:
            rep, rnn_hidden_states = self.state_encoder(
                rep, memory.tensor("rnn"), masks
            )

        distributions = (
            CategoricalDistr(logits=self.actor(rep))
            if not self.is_multi_discrete
            else (
                MultiCategorical(
                    logits=self.actor(rep), action_dim=self.action_space.nvec.tolist()
                )
            )
        )
        # distributions = self.actor(rep)
        return (
            ActorCriticOutput(
                distributions=distributions, values=self.critic(rep), extras={}
            ),
            memory.set_tensor("rnn", rnn_hidden_states) if self.use_rnn else None,
        )


VISUAL_BACKBONES = dict(cnn=SimpleCNN, resnet=ResNetEncoder)
