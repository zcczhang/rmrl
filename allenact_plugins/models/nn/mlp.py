from functools import partial
from typing import Optional, Callable, Union, Tuple, Sequence

import numpy as np
import torch
from torch import nn


class MLPActorCritic(nn.Module):
    """MLP ActorCritic.

    # Attributes
        state_dim: state obs dim
        action_dim : dim of action dim for prediction
        env_action_dim: "real" action dim for `prev_actions` from env,
            different from above if is multi-discrete action space)
        hidden_dim : tuple or int of actor-critic hidden dimension.
        extra_feature_dim: extra feature dim (e.g. visual embedding) to concatenate in `forward`
        num_stacked_frames: num stacked state obs for multiplying state_dim
        activation_fn: activation function across fc layers
        shared_state_encoder: using one shared state encoder for state obs before passing to actor & critic
        separate_state_encoder: separate state encoder for state obs while passing to actor & critic
        critic_condition_action: whether concatenate/conditioned on actions for critic (more for off-policy)
        use_action_embed: whether using embedding when critic conditioned on (multi-discrete) action
        action_embed_dim: embedding dim if using action embedding
        obs_feature_dim: output dim for state obs encoder if using
        squash_output: whether using tanh() after last linear layer
        orthogonal_init: whether orthogonal initialization
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env_action_dim: Optional[int] = None,
        hidden_dim: Union[Sequence[int], int] = (512, 256),
        extra_feature_dim: Optional[int] = None,
        num_stacked_frames: Optional[int] = None,
        activation_fn: Callable = nn.Tanh,
        shared_state_encoder: bool = True,
        separate_state_encoder: bool = False,
        critic_condition_action: bool = False,
        obs_feature_dim: int = 512,
        use_action_embed: bool = True,
        action_embed_dim: Optional[int] = 32,
        squash_output: bool = True,
        orthogonal_init: bool = False,
    ):
        super().__init__()
        if separate_state_encoder:
            assert (
                not shared_state_encoder
            ), "there could be no state encoder but cannot be both shared or separate"
            assert obs_feature_dim is not None
        env_action_dim = env_action_dim or action_dim
        self._state_encoder = one_layer_state_encoder(
            state_dim=state_dim,
            output_size=obs_feature_dim if shared_state_encoder else None,
            num_stacked_frames=num_stacked_frames,
        )
        extra_feature_dim = extra_feature_dim or 0
        mlp_in_features = (
            obs_feature_dim
            if (shared_state_encoder or separate_state_encoder)
            else state_dim * num_stacked_frames  # no 1-layer state encoder
        )
        hidden_dim = (hidden_dim,) if isinstance(hidden_dim, int) else hidden_dim
        mlp_hidden_dim = (mlp_in_features + extra_feature_dim,) + tuple(hidden_dim)

        self._actor = nn.Sequential()
        if separate_state_encoder:
            self._actor.append(
                one_layer_state_encoder(
                    state_dim=state_dim,
                    output_size=obs_feature_dim,
                    num_stacked_frames=num_stacked_frames,
                )
            )
        for mod in make_mlp_hidden(activation_fn, *mlp_hidden_dim):
            self._actor.append(mod)
        self._actor_last = nn.Linear(mlp_hidden_dim[-1], action_dim)
        self._actor_act = nn.Tanh() if squash_output else nn.Identity()

        self._critic_state_encoder = (
            one_layer_state_encoder(
                state_dim=state_dim,
                output_size=obs_feature_dim,
                num_stacked_frames=num_stacked_frames,
            )
            if separate_state_encoder
            else nn.Identity()
        )

        self._critic_condition_action = critic_condition_action
        if critic_condition_action:
            if use_action_embed:
                assert action_embed_dim is not None
                self._action_embedding = nn.Embedding(action_dim, action_embed_dim)
            else:
                action_embed_dim = 1  # multiply env_action_dim below
                self._action_embedding = nn.Identity()
            # input (env) action dim * action embed dims
            mlp_hidden_dim = (
                mlp_in_features + extra_feature_dim + action_embed_dim * env_action_dim,
            ) + tuple(hidden_dim)
        self._critic = nn.Sequential(
            *make_mlp_hidden(activation_fn, *mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim[-1], 1),
        )
        self._last_hidden_dim = mlp_hidden_dim[-1]
        if orthogonal_init:
            orthogonal_map = {
                self._actor: 0.01,  # np.sqrt(2)
                self._actor_last: 0.01,
                self._critic: 1.0,
            }
            for mod, gain in orthogonal_map.items():
                mod.apply(partial(orthogonal_init_fn, gain=gain))

    @property
    def last_hidden_dim(self):
        return self._last_hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        *,
        prev_actions: Optional[torch.Tensor] = None,
        extra_rep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """return last hidden states for gSDE distribution.

        return: actor rep, critic rep, actor_last_hidden (for gSDE)
        """
        x = x.view(*x.shape[:2], -1)
        x = self._state_encoder(x)
        if extra_rep is not None:
            x = torch.cat([x, extra_rep], dim=-1)
        actor_last_hidden = self._actor(x)
        means = self._actor_act(self._actor_last(actor_last_hidden))
        if self._critic_condition_action:
            assert prev_actions is not None
            action_embed = self._action_embedding(prev_actions)
            values = self._critic(
                torch.cat(
                    [
                        self._critic_state_encoder(x),
                        action_embed.view(*action_embed.shape[:2], -1),
                    ],
                    dim=-1,
                )
            )
        else:
            values = self._critic(self._critic_state_encoder(x))
        return means, values, actor_last_hidden


def one_layer_state_encoder(
    state_dim: int,
    output_size: Optional[int] = None,
    num_stacked_frames: Optional[int] = None,
    add_layernorm: bool = True,
    activation_fn: nn.Module = nn.Tanh(),
) -> Optional[Union[torch.nn.Sequential, nn.Identity]]:
    if output_size is None:
        return nn.Identity()
    if state_dim is None or state_dim == 0:
        return None
    enc = nn.Sequential(nn.Linear(state_dim * (num_stacked_frames or 1), output_size))
    if add_layernorm:
        enc.append(nn.LayerNorm(output_size))
    enc.append(activation_fn)
    return enc


def make_mlp_hidden(nl: Callable, *dims) -> list:
    """create mlp hidden layers list."""
    res = []
    for it, dim in enumerate(dims[:-1]):
        res.append(nn.Linear(dim, dims[it + 1]))
        res.append(nl())
    return res


def orthogonal_init_fn(mod: nn.Module, gain: Union[float, int] = np.sqrt(2)):
    """For orthogonal initialization."""
    if isinstance(mod, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(mod.weight, gain=gain)
        if mod.bias is not None:
            mod.bias.data.fill_(0.0)
