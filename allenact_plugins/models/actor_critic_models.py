from functools import partial
from typing import Dict, Union, Optional, Tuple, Any, Sequence, cast, Literal

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
)
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact_plugins.distributions import (
    POSSIBLE_DIST,
    StateDependentNoiseDistribution,
)
from allenact_plugins.models.nn.mlp import make_mlp_hidden, orthogonal_init_fn


class ActorCriticContinuous(ActorCriticModel[Normal]):
    """shared ActorCriticModel in continuous space.

    # Attributes

    input_uuid : observation sensor input uuid,
    action_space : env continuous action space,
    observation_space : env observation space,
    num_stacked_frames : num of frame stacking of observations,
    distribution_type : Distribution name
    gSDE_sample_freq: gSDE sample (reset noise) frequencies if using gSDE (-1 for only once initially)
    use_log_std : if true, the parametrized std is log_std, exp() before passing to the distribution
    min_action_std : min std/log_std
    max_action_std : max std/log_std
    use_rnn : use rnn encoder if true, otherwise use MLP policy
    rnn_hidden_size : rnn hidden size, None for MLP
    trainable_masked_hidden_state : rnn arg
    num_rnn_layers : rnn arg
    rnn_type : rnn arg, LSTM or GRU
    mlp_hidden_dims : MLP dimension if using MLP policy
    activation_fn : activation function
    squash_output : whether to squash the output using a Tanh
    orthogonal_init : whether orthogonal initialize layers
    use_phase_embed : whether using phase embedding for two (or more) phases training
    phase_input_uuid : phase sensor input uuid
    phase_embed_dim : categorical phase embedding dimension
    single_policy : whether using separate actor & critic heads for two phases trainings
    use_obs_mlp_encode : whether using a shared linear encoder to encode the obs first
    logging_std : whether logging the mean value of std in metrics
    """

    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Box,
        observation_space: gym.spaces.Dict,
        num_stacked_frames: Optional[int] = 5,
        distribution_type: Literal[
            "TruncatedNormal", "GaussianDistr", "sGDE"
        ] = "GaussianDistr",
        gSDE_sample_freq: int = 4,
        use_log_std: bool = False,
        min_action_std: float = 0.1,
        max_action_std: float = 0.8,
        use_rnn: bool = False,
        rnn_hidden_size: Optional[int] = None,  # 512
        trainable_masked_hidden_state: bool = False,
        num_rnn_layers: int = 2,
        rnn_type: str = "GRU",  # LSTM or GRU
        mlp_hidden_dims: Sequence[int] = (512, 256),
        activation_fn: str = "tanh",
        squash_output: bool = True,
        orthogonal_init: bool = False,
        use_phase_embed: bool = False,
        phase_input_uuid: Optional[str] = "phase_sensor",
        phase_embed_dim: int = 4,
        num_phases: int = 2,
        single_policy: bool = True,
        use_obs_mlp_encode: bool = False,
        logging_std: bool = True,
        **mlp_kwargs,
    ):
        super().__init__(action_space, observation_space)
        if use_rnn:
            # make sure rnn_hidden_size is not None
            rnn_hidden_size = rnn_hidden_size or 512

        # ======== select activation function ========
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation_fn]

        # ======== select distribution ========
        self.dist = POSSIBLE_DIST[distribution_type]
        self.distribution_type = distribution_type
        # sGDE only support log std so far
        self.use_log_std = True if distribution_type == "sGDE" else use_log_std

        # ======== init dim attributes ========
        self.input_uuid = input_uuid
        assert len(observation_space[self.input_uuid].shape) == 1
        state_dim = observation_space[self.input_uuid].shape[0]
        assert len(action_space.shape) == 1
        self.action_dim = action_space.shape[0]

        # ======== phase embedding ========
        phase_embed_dim = phase_embed_dim if use_phase_embed else 0
        self.use_phase_embed = use_phase_embed
        phase_dim = 0
        if use_phase_embed:
            self.num_phases = num_phases
            assert isinstance(observation_space[phase_input_uuid], gym.spaces.Discrete)
            phase_dim = observation_space[phase_input_uuid].n
            # input: phase dim, output: phase_dim * phase_embed_dim
            self.phase_embed = nn.Embedding(self.num_phases, phase_embed_dim)

        #  ======== model arch & dim init ========
        self.rnn_hidden_size = rnn_hidden_size
        self.use_obs_mlp_encode = use_obs_mlp_encode
        self.use_rnn = use_rnn
        # affect model input dim, replace None num_stacked_frames case with 1
        num_stacked_frames = num_stacked_frames or 1
        if use_rnn:
            # init recurrent encoder
            input_size = state_dim * num_stacked_frames + phase_embed_dim * phase_dim
            self.state_encoder = RNNStateEncoder(
                input_size=input_size,
                hidden_size=rnn_hidden_size,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
            )
        else:
            if use_obs_mlp_encode:
                # a shared linear observation encoder
                obs_mlp_dims = (
                    state_dim * num_stacked_frames,
                    mlp_hidden_dims[0],
                )
                self.mlp_encoder = nn.Sequential(
                    *make_mlp_hidden(activation_fn, *obs_mlp_dims)
                )
                # mlp_hidden_dims for actor critic heads
                mlp_hidden_dims = (
                    mlp_hidden_dims[0] + phase_embed_dim * phase_dim,
                ) + tuple(mlp_hidden_dims)
            else:
                # mlp_hidden_dims for actor critic heads
                mlp_hidden_dims = (
                    state_dim * num_stacked_frames + phase_embed_dim * phase_dim,
                ) + tuple(mlp_hidden_dims)

        # ======== init MLP actor and critic ========
        ac_input_dim = rnn_hidden_size if use_rnn else mlp_hidden_dims
        self.actor_critic = MLPActorCritic(
            hidden_dim=ac_input_dim,
            action_dim=self.action_dim,
            activation_fn=activation_fn,
            squash_output=squash_output,
            orthogonal_init=orthogonal_init,
        )

        # ======== parameterized std for distributions ========
        std_init = min_action_std if self.use_log_std else 0.0
        if distribution_type == "gSDE":
            # initialize gSDE
            self.dist = self.dist(
                latent_sde_dim=mlp_hidden_dims[-1] if not use_rnn else num_rnn_layers,
                action_dim=self.action_dim,
                log_std_init=std_init,
                use_expln=False,
                learn_features=False,
            )
            self.std = self.dist.log_std
        else:
            self.std = nn.Parameter(
                torch.ones(self.action_dim, dtype=torch.float32) * std_init,
                requires_grad=True,
            )
        self.min_action_std = min_action_std
        self.max_action_std = max_action_std

        # ======== separate actor-critic heads for dual policies ========
        self.single_policy = single_policy
        if not single_policy:
            # Note: use single std parameters for two policies
            self.actor_critic_1 = MLPActorCritic(
                hidden_dim=ac_input_dim,
                action_dim=self.action_dim,
                activation_fn=activation_fn,
                squash_output=squash_output,
                orthogonal_init=orthogonal_init,
            )

        # ======== orthogonal initialization for mlp encoder if using ========
        if orthogonal_init and use_obs_mlp_encode:
            orthogonal_map = {self.mlp_encoder: np.sqrt(2)}
            for mod, gain in orthogonal_map.items():
                mod.apply(partial(orthogonal_init_fn, gain=gain))

        # ======== other attributes ========
        self.phase_input_uuid = phase_input_uuid
        self.logging_std = logging_std
        if self.distribution_type == "gSDE":
            self.gSDE_sample_freq = gSDE_sample_freq

    def forward(  # type:ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        # shapes
        nsteps, nsamplers, nagents = masks.shape[:3]

        # 1. obs representation
        # raw observation
        obs_rep = observations[self.input_uuid].view(nsteps, nsamplers * nagents, -1)
        if self.use_obs_mlp_encode:
            # encode obs first
            obs_rep = self.mlp_encoder(obs_rep)

        # 2. representation with phase embedding if using
        if self.use_phase_embed:
            # current phase (step, sampler, phase_dim)
            phase = cast(Dict[str, torch.LongTensor], observations)[
                self.phase_input_uuid
            ]
            phase_rep = self.phase_embed(phase).view(nsteps, nsamplers * nagents, -1)
            # concatenate phase and obs
            rep = torch.cat([obs_rep, phase_rep], dim=-1)
        else:
            rep = obs_rep

        # 3. encode RNN if using
        if self.use_rnn:
            rep, rnn_hidden_states = self.state_encoder(
                rep, memory.tensor("rnn"), masks
            )
        else:
            rnn_hidden_states = None

        # 4. pass actor & critic
        mean, values, actor_last_hidden = self.actor_critic(rep)

        # 5. double policies case
        if not self.single_policy:
            phase = cast(Dict[str, torch.LongTensor], observations)[
                self.phase_input_uuid
            ]
            mean_1, values_1, actor_last_hidden_1 = self.actor_critic(rep)

            mean = (1 - phase) * mean + phase * mean_1
            values = (1 - phase) * values + phase * values_1
            actor_last_hidden = (
                1 - phase
            ) * actor_last_hidden + phase * actor_last_hidden_1

        # 6. constraint std
        std = (
            self.std
            if self.use_log_std
            else (
                self.min_action_std
                + (self.max_action_std - self.min_action_std) * torch.sigmoid(self.std)
            )
        )

        if self.use_log_std:
            std = torch.exp(std)

        # 7. get distribution and returns
        distribution = (
            self.dist(loc=mean, scale=std)
            if self.distribution_type != "gSDE"
            else self.dist.distribution(loc=mean, latent_sde=actor_last_hidden)
        )
        extras = {"std": std.detach().clone().mean()} if self.logging_std else {}
        return (
            ActorCriticOutput(
                distributions=cast(DistributionType, distribution),
                values=values,
                extras=extras,
            ),
            memory.set_tensor("rnn", rnn_hidden_states) if self.use_rnn else None,
        )

    @property
    def recurrent_hidden_state_size(self) -> Optional[int]:
        """The recurrent hidden state size of the model, None for MLP."""
        return self.rnn_hidden_size if self.use_rnn else None

    @property
    def num_recurrent_layers(self) -> Optional[int]:
        """Number of recurrent hidden layers, None for MLP."""
        return self.state_encoder.num_recurrent_layers if self.use_rnn else None

    def _recurrent_memory_specification(self) -> Optional[dict]:
        """The memory specification for the `ActorCriticModel`. See docs for
        `_recurrent_memory_shape`

        # Returns

        The memory specification from `_recurrent_memory_shape` if using RNN else None.
        """
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

    def reset_noise(self, num_env: int):
        """Sample a new noise matrix every n steps when using gSDE."""
        assert isinstance(self.dist, StateDependentNoiseDistribution)
        self.dist.sample_weights(self.std, num_env)


class ActorCriticContinuousIrrHead(ActorCriticContinuous):
    """model with irreversible distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.distribution_type != "gSDE"
        ), "gSDE for irreversible checking model is not implemented so far"
        self.actor_critic.init_irr_head()

    def forward(
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        ac_output, mem = super().forward(observations, memory, prev_actions, masks)
        irr_means = self.actor_critic.irr_means
        std = (
            self.std
            if self.use_log_std
            else (
                self.min_action_std
                + (self.max_action_std - self.min_action_std) * torch.sigmoid(self.std)
            )
        )
        if self.use_log_std:
            std = torch.exp(std)
        ac_output.extras["irr_distributions"] = self.dist(loc=irr_means, scale=std)
        return ac_output, mem


# TODO refactor from nn.mlp class
class MLPActorCritic(nn.Module):
    """MLP ActorCritic.

    Attributes:
        hidden_dim : tuple or int of actor dimension. int for linear actor
        action_dim : output dim
        activation_fn : activation function across fc layers
        squash_output : whether using tanh() after last linear layer
        orthogonal_init : whether orthogonal initialization
    """

    def __init__(
        self,
        hidden_dim: Union[tuple, int],
        action_dim: int,
        activation_fn: nn.Module = nn.Tanh,
        squash_output: bool = True,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        if isinstance(hidden_dim, int):
            self._actor = nn.Identity()
            self._critic = nn.Linear(hidden_dim, 1)
            hidden_dim = (hidden_dim,)
        else:
            self._actor = nn.Sequential(*make_mlp_hidden(activation_fn, *hidden_dim))
            self._critic = nn.Sequential(
                *make_mlp_hidden(activation_fn, *hidden_dim),
                nn.Linear(hidden_dim[-1], 1),
            )
        self.last_hidden_dim = hidden_dim[-1]
        self.action_dim = action_dim
        self._actor_last = nn.Linear(hidden_dim[-1], action_dim)
        self._act = nn.Tanh() if squash_output else nn.Identity()

        if orthogonal_init:
            orthogonal_map = {
                self._actor: 0.01,  # np.sqrt(2)
                self._actor_last: 0.01,
                self._critic: 1.0,
            }
            for mod, gain in orthogonal_map.items():
                mod.apply(partial(orthogonal_init_fn, gain=gain))

        self.squash_output = squash_output
        # for irreversible checking head
        self.irr_head = None
        self.irr_means = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return last hidden states for gSDE distribution.

        return: actor rep, critic rep, actor_last_hidden (for gSDE)
        """
        actor_last_hidden = self._actor(x)
        values = self._critic(x)
        means = self._act(self._actor_last(actor_last_hidden))

        if self.irr_head is not None:
            self.irr_means = self.irr_head(actor_last_hidden)

        return means, values, actor_last_hidden

    def init_irr_head(self):
        # irreversible checking head
        self.irr_head = nn.Linear(self.last_hidden_dim, self.action_dim)
