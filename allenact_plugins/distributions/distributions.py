from collections import OrderedDict
from typing import Dict, Optional, Union, List, Tuple

import torch
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal

from allenact.base_abstractions.distributions import Distr, CategoricalDistr


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Continuous actions are usually considered to be independent, so we can
    sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    ref: stable-baselines3
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class GaussianDistr(torch.distributions.Normal, Distr):
    """PyTorch's Normal distribution with a `mode` method."""

    def mode(self) -> torch.FloatTensor:
        return super().mean

    def enumerate_support(self, expand: bool = True):
        raise NotImplementedError


class StateDependentNoiseDistribution(Distr):
    """Distribution class for using generalized State Dependent Exploration
    (gSDE).

    Paper: https://arxiv.org/abs/2005.05719
    Modified from: https://github.com/DLR-RM/stable-baselines3/blob/646d6d38b6ba9aac612d4431176493a465ac4758/stable_baselines3/common/distributions.py#L406
    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    # Attributes:
        latent_sde : out features of last hidden layer
        action_dim: Dimension of the action space.
        log_std_init: log_std initial value
        use_expln: Use ``expln()`` function instead of ``exp()`` to ensure a positive
                   stddev (cf paper). It allows to keep variance above zero and prevent
                   it from growing too fast. In practice, ``exp()`` is usually enough.
        learn_features : Whether to learn features for gSDE or not.

        TODO not worked
    """

    def __init__(
        self,
        latent_sde_dim: int,
        action_dim: int,
        log_std_init: Union[float, int] = 0,
        use_expln: bool = True,
        learn_features: bool = True,
    ):
        self.latent_sde_dim = latent_sde_dim
        self.action_dim = action_dim
        self.use_expln = use_expln
        self.learn_features = learn_features

        self._latent_sde = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None

        self._log_std = torch.nn.Parameter(
            torch.ones(latent_sde_dim, action_dim, dtype=torch.float32) * log_std_init,
            requires_grad=True,
        )
        # self.sample_weights(self._log_std)

        self.dist: Normal = None  # type: ignore

    @property
    def log_std(self):
        return self._log_std

    def mode(self):
        return self.dist.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def sample_weights(
        self, log_std: torch.Tensor, batch_size: int = 1, pass_gradients: bool = False
    ) -> None:
        """Sample weights for the noise exploration matrix, using a centered
        Gaussian distribution."""
        std = self.get_std(log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        # Re-parametrization trick to pass gradients
        self.exploration_mat = (
            self.weights_dist.rsample()
            if pass_gradients
            else self.weights_dist.sample()
        )
        # Pre-compute matrices in case of parallel exploration
        # TODO TypeError: torch.Size() takes an iterable of 'int' (item 0 is 'tuple')
        # rsample((batch_size, ))
        self.exploration_matrices = self.weights_dist.rsample()

    def distribution(
        self, loc: torch.Tensor, latent_sde: torch.Tensor
    ) -> "StateDependentNoiseDistribution":
        """update distribution with input loc and last hidden state."""
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = self._latent_sde ** 2 @ self.get_std(self.log_std) ** 2
        self.dist = Normal(loc=loc, scale=torch.sqrt(variance + 1e-6))
        return self

    def get_std(self, log_std: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = torch.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + epsilon
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            std = torch.exp(log_std)
        return std

    def sample(self, sample_shape=torch.Size()):
        noise = self.get_noise(self._latent_sde)
        actions = self.dist.mean + noise
        return actions

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return latent_sde @ self.exploration_mat
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = torch.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def enumerate_support(self, expand: bool = True):
        raise NotImplementedError


class TruncatedNormal(Normal, Distr):
    """Truncated Normal distribution:

    bounding the random variable from either below or above (or both)
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(
        self,
        clip: Optional[torch.Tensor] = None,
        sample_shape: torch.Size = torch.Size(),
    ):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

    def mode(self) -> torch.FloatTensor:
        return super().mean

    def enumerate_support(self, expand: bool = True):
        raise NotImplementedError


class MultiCategorical(torch.distributions.Distribution, Distr):
    def __init__(self, logits, action_dim: List[int]):
        super().__init__()
        self._dists = [
            CategoricalDistr(logits=split)
            for split in torch.split(logits, action_dim, dim=-1)
        ]

    def log_prob(self, actions):
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=-1).sum(dim=-1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=-1)

    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=-1) for dist in self._dists], dim=-1
        )


class DictActionSpaceDistr(Distr):
    def __init__(
        self,
        cont_distr: Tuple[str, GaussianDistr],
        disc_distr: Tuple[str, CategoricalDistr],
        ordered_keys: list,
    ):
        """continuous (Gaussian) distribution, discrete (Categorical)
        distribution."""
        self.cont_key, self.cont_distr = cont_distr
        self.disc_key, self.disc_distr = disc_distr
        assert self.cont_key in ordered_keys
        assert self.disc_key in ordered_keys
        self.ordered_keys = ordered_keys
        self.disc_first = ordered_keys[0] == self.disc_key
        self.ordered_dists = OrderedDict(
            {
                ordered_keys[0]: self.cont_distr
                if not self.disc_first
                else self.disc_distr,
                ordered_keys[1]: self.cont_distr
                if self.disc_first
                else self.disc_distr,
            }
        )

    def log_prob(self, actions: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.cont_distr.log_prob(
            actions[self.cont_key]
        ) + self.disc_distr.log_prob(actions[self.disc_key].unsqueeze(-1))

    def entropy(self):
        return self.cont_distr.entropy() + self.disc_distr.entropy().unsqueeze(-1)

    def sample(self, sample_shape=torch.Size()):
        dict_action = OrderedDict()
        for key in self.ordered_keys:
            dict_action[key] = self.ordered_dists[key].sample(sample_shape)
        return dict_action

    def mode(self):
        dict_action = OrderedDict()
        for key in self.ordered_keys:
            dict_action[key] = self.ordered_dists[key].mode()
        return dict_action
