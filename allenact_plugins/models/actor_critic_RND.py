from typing import Dict, Union, Optional, Tuple, Any, Sequence, cast, Literal

import gym
import numpy as np
import torch
import torch.nn as nn

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
)
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact_plugins.distributions import GaussianDistr
from allenact_plugins.models.actor_critic_models import make_mlp_hidden, POSSIBLE_DIST


class ActorCriticRND(ActorCriticModel[GaussianDistr]):
    def __init__(
        self,
        input_uuid: str,
        action_space: gym.spaces.Box,
        observation_space: gym.spaces.Dict,
        num_stacked_frames: int = 5,
        distribution_type: Literal[
            "TruncatedNormal", "GaussianDistr"
        ] = "GaussianDistr",
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
        orthogonal_init: bool = True,
        use_phase_embed: bool = False,
        phase_input_uuid: str = "phase_sensor",
        phase_embed_dim: int = 2,
        single_policy: bool = True,
        use_obs_mlp_encode: bool = False,
        logging_std: bool = False,
        # for intrinsic rewards
        use_gae: bool = True,
        gamma: float = 0.999,
        tau: float = 0.95,
    ):
        super().__init__(action_space, observation_space)
        if use_rnn:
            rnn_hidden_size = rnn_hidden_size or 512

        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation_fn]
        assert not use_obs_mlp_encode, (
            f"not support MLP obs encoder for now, "
            f"just pass to be consistent with ActorCriticContinuous Model"
        )

        self.input_uuid = input_uuid
        assert len(observation_space[self.input_uuid].shape) == 1
        self.state_dim = observation_space[self.input_uuid].shape[0]
        assert len(action_space.shape) == 1
        self.action_dim = action_space.shape[0]

        assert num_stacked_frames >= 2, "need to at least 2 frame stacking for RND"

        phase_embed_dim = phase_embed_dim if use_phase_embed else 0
        self.use_phase_embed = use_phase_embed
        if use_phase_embed:
            self.num_phases = 2
            self.phase_embed = nn.Embedding(self.num_phases, phase_embed_dim)
        self.single_policy = single_policy

        self.rnn_hidden_size = rnn_hidden_size
        self.use_rnn = use_rnn
        if use_rnn:
            input_size = self.state_dim * (num_stacked_frames - 1) + phase_embed_dim
            self.state_encoder = RNNStateEncoder(
                input_size=input_size,
                hidden_size=self.rnn_hidden_size,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
            )
            self.actor = nn.Linear(rnn_hidden_size, self.action_dim)

            # external & internal critic
            self.critic_shared = nn.Linear(rnn_hidden_size, rnn_hidden_size)
            self.ext_critic_last_layer = nn.Linear(rnn_hidden_size, 1)
            self.int_critic_last_layer = nn.Linear(rnn_hidden_size, 1)

            # predictor and target net take an observation to an embedding
            self.predictor = nn.Linear(rnn_hidden_size, rnn_hidden_size)
            self.target = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        else:
            mlp_hidden_dims = (
                self.state_dim * (num_stacked_frames - 1) + phase_embed_dim,
            ) + tuple(mlp_hidden_dims)
            self.actor = nn.Sequential(
                *make_mlp_hidden(activation_fn, *mlp_hidden_dims),
                nn.Linear(mlp_hidden_dims[-1], self.action_dim),
                nn.Tanh() if squash_output else nn.Identity,
            )

            # external & internal critic
            self.critic_shared = nn.Sequential(
                nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
                nn.Tanh(),
                nn.Linear(mlp_hidden_dims[1], mlp_hidden_dims[-1]),
                nn.Tanh(),
            )
            self.ext_critic_last_layer = nn.Linear(mlp_hidden_dims[-1], 1)
            self.int_critic_last_layer = nn.Linear(mlp_hidden_dims[-1], 1)

            # predictor and target net take an observation to an embedding
            self.predictor = nn.Sequential(
                *make_mlp_hidden(activation_fn, *mlp_hidden_dims),
                nn.Linear(mlp_hidden_dims[-1], mlp_hidden_dims[-1]),
            )
            self.target = nn.Sequential(
                *make_mlp_hidden(activation_fn, *mlp_hidden_dims),
                nn.Linear(mlp_hidden_dims[-1], mlp_hidden_dims[-1]),
            )

        self.std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32))
        self.min_action_std = min_action_std
        self.max_action_std = max_action_std
        self.use_log_std = use_log_std

        self.int_r_kwargs = dict(use_gae=use_gae, gamma=gamma, tau=tau)

        if not single_policy:
            # separate policy for phase 1
            self.actor_1 = (
                nn.Linear(rnn_hidden_size, self.action_dim)
                if use_rnn
                else nn.Sequential(
                    *make_mlp_hidden(activation_fn, *mlp_hidden_dims),
                    nn.Linear(mlp_hidden_dims[-1], self.action_dim),
                    nn.Tanh() if squash_output else nn.Identity(),
                )
            )

            self.ext_critic_1_last_layer = nn.Linear(
                rnn_hidden_size if self.use_rnn else mlp_hidden_dims[-1], 1
            )
            self.std_1 = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.float32))

        if orthogonal_init:
            # follow the init of original implementation of RND
            init_list = [self.actor, self.critic_shared, self.predictor, self.target]
            if not single_policy:
                init_list.append(self.actor_1)

            for mod in init_list:
                if isinstance(mod, nn.Linear):
                    nn.init.orthogonal_(mod.weight, gain=np.sqrt(2))
                    mod.bias.data.zero_()
                else:
                    for layer in mod:
                        if isinstance(layer, nn.Linear):
                            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                            layer.bias.data.zero_()

            critic_layers = [self.ext_critic_last_layer, self.int_critic_last_layer]
            if not single_policy:
                critic_layers.append(self.ext_critic_1_last_layer)
            for layer in critic_layers:
                nn.init.orthogonal_(layer.weight, gain=0.01)
                layer.bias.data.zero_()

        # fixed and randomly initialized target network
        for param in self.target.parameters():
            param.requires_grad = False

        # other attributes
        self.distribution_type = distribution_type
        self.phase_input_uuid = phase_input_uuid
        self.logging_std = logging_std

    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self.rnn_hidden_size if self.use_rnn else None

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers if self.use_rnn else None

    def _recurrent_memory_specification(self):
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

    def forward(  # type:ignore
        self,
        observations: Dict[str, Union[torch.FloatTensor, Dict[str, Any]]],
        memory: Memory,
        prev_actions: Any,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        nsteps, nsamplers, nagents = masks.shape[:3]
        # if obs is img, it can be encoded first
        obs = observations[self.input_uuid].view(nsteps, nsamplers * nagents, -1)
        # state and next_obs based on the stacked frames
        state = obs[:, :, self.state_dim :]
        next_obs = obs[:, :, : -self.state_dim]

        phase = None
        if self.use_phase_embed:
            phase = cast(Dict[str, torch.LongTensor], observations)[
                self.phase_input_uuid
            ]
            phase_rep = self.phase_embed(phase).view(nsteps, nsamplers * nagents, -1)
            rep = torch.cat([state, phase_rep], dim=-1)
            next_obs = torch.cat([next_obs, phase_rep], dim=-1)
        else:
            rep = state

        if self.use_rnn:
            rep, rnn_hidden_states = self.state_encoder(
                rep, memory.tensor("rnn"), masks
            )
            next_obs_out, _ = self.state_encoder(next_obs, memory.tensor("rnn"), masks)

            values_external_0 = self.ext_critic_last_layer(
                self.critic_shared(rep) + rep
            )
            values_internal = self.int_critic_last_layer(self.critic_shared(rep) + rep)
        else:
            rep, rnn_hidden_states, next_obs_out = rep, None, next_obs

            values_external_0 = self.ext_critic_last_layer(self.critic_shared(rep))
            values_internal = self.int_critic_last_layer(self.critic_shared(rep))

        # policy
        mean_0 = self.actor(rep)
        std_0 = self.min_action_std + (
            self.max_action_std - self.min_action_std
        ) * torch.sigmoid(self.std)

        pred_embed = cast(torch.FloatTensor, self.predictor(next_obs_out))
        with torch.no_grad():
            target_embed = cast(torch.FloatTensor, self.target(next_obs_out))

        # self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        intrinsic_reward = (
            (target_embed - pred_embed).pow(2).mean(-1).reshape(nsteps, nsamplers, 1)
        )

        if not self.single_policy:
            if phase is None:
                phase = cast(Dict[str, torch.LongTensor], observations)[
                    self.phase_input_uuid
                ]
            mean_1 = self.actor_1(rep)

            values_1 = (
                self.ext_critic_1_last_layer(self.critic_shared(rep) + rep)
                if self.use_rnn
                else self.ext_critic_1_last_layer(self.critic_shared(rep))
            )

            std_1 = self.min_action_std + (
                self.max_action_std - self.min_action_std
            ) * torch.sigmoid(self.std_1)

            mean = (1 - phase) * mean_0 + phase * mean_1
            values_external = (1 - phase) * values_external_0 + phase * values_1
            std = (1 - phase) * std_0 + phase * std_1
        else:
            mean = mean_0
            values_external = values_external_0
            std = std_0

        if self.use_log_std:
            std = std.exp()

        extras = dict(
            values_internal=values_internal,
            intrinsic_reward=intrinsic_reward,
            # for calculating intrinsic advantage in loss
            int_r_kwargs=self.int_r_kwargs,
        )
        if self.logging_std:
            extras["std"] = std.detach().clone().mean()
        return (
            ActorCriticOutput(
                distributions=cast(
                    DistributionType,
                    POSSIBLE_DIST[self.distribution_type](loc=mean, scale=std),
                ),
                values=values_external,
                extras=extras,
            ),
            memory.set_tensor("rnn", rnn_hidden_states) if self.use_rnn else None,
        )
