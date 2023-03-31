from typing import Dict, Optional, Tuple, Sequence, Union, Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from allenact.base_abstractions.sensor import SpaceDict
from allenact.utils.model_utils import make_cnn, compute_cnn_output
from allenact_plugins.models.nn.visual_base import VisualBase


class SimpleCNN(VisualBase):
    """A Simple N-Conv CNN followed by a fully connected layer. Takes in
    observations (of type gym.spaces.dict) and produces an embedding of the
    `rgb_uuid`

    # Attributes
    observation_space : The observation_space of the agent, should have `rgb_uuid` or `depth_uuid` as
        a component (otherwise it is a blind model).
    output_size : The size of the embedding vector to produce.
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        *,
        output_size: int,
        # use if `forward` input is dict for value with key of `rgb_uuid`
        rgb_uuid: Optional[str],
        num_stacked_frames: Optional[int] = None,
        channel_first: bool = True,
        # fc followed by CNN
        flatten: bool = True,
        output_relu: bool = False,
        output_layer_norm: bool = True,
        # CNN config (has to be the same length)
        layer_channels: Sequence[int] = (32, 64, 32),
        # could be tuple or int. e.g. ((3, 3), (3, 3)) is equivalent to (3, 3))
        kernel_sizes: Sequence[Union[Tuple[int, int], int]] = ((8, 8), (4, 4), (3, 3)),
        layers_stride: Sequence[Union[Tuple[int, int], int]] = ((4, 4), (2, 2), (1, 1)),
        paddings: Sequence[Union[Tuple[int, int], int]] = ((0, 0), (0, 0), (0, 0)),
        dilations: Sequence[Union[Tuple[int, int], int]] = ((1, 1), (1, 1), (1, 1)),
        # augmentation fn and its kwargs
        augmentation: Optional[Union[str, Callable]] = None,
        augmentation_kwargs: Optional[dict] = None,
    ):
        if output_layer_norm:
            assert not output_relu and flatten
        self._cnn_layers_channels = list(layer_channels)
        self._cnn_layers_kernel_size = list(kernel_sizes)
        self._cnn_layers_stride = list(layers_stride)
        self._cnn_layers_paddings = list(paddings)
        self._cnn_layers_dilations = list(dilations)
        super().__init__(
            observation_space=observation_space,
            output_size=output_size,
            rgb_uuid=rgb_uuid,
            num_stacked_frames=num_stacked_frames,
            channel_first=channel_first,
            flatten=flatten,
            output_relu=output_relu,
            output_layer_norm=output_layer_norm,
        )
        aug = AUG_MAP[augmentation] if isinstance(augmentation, str) else augmentation
        self.aug = aug(**(augmentation_kwargs or {})) if augmentation else nn.Identity()

    @staticmethod
    def _maybe_int2tuple(*args) -> List[Tuple[int, int]]:
        return [(_, _) if isinstance(_, int) else _ for _ in args]

    def setup_model(
        self,
        output_size: int,
        input_dims: np.ndarray,
        input_channels: int,
        flatten: bool = True,
        output_relu: bool = True,
        output_layer_norm: bool = False,
        **kwargs,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
            self._cnn_layers_kernel_size,
            self._cnn_layers_stride,
            self._cnn_layers_paddings,
            self._cnn_layers_dilations,
        ):
            kernel_size, stride, padding, dilation = self._maybe_int2tuple(
                kernel_size, stride, padding, dilation
            )
            output_dims = self._conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )
        cnn = make_cnn(
            input_channels=input_channels,
            layer_channels=self._cnn_layers_channels,
            kernel_sizes=self._cnn_layers_kernel_size,
            strides=self._cnn_layers_stride,
            paddings=self._cnn_layers_paddings,
            dilations=self._cnn_layers_dilations,
            output_height=output_dims[0],
            output_width=output_dims[1],
            output_channels=output_size,
            flatten=flatten,
            output_relu=output_relu,
        )
        if output_layer_norm:
            cnn.add_module("layer_norm", nn.LayerNorm(output_size))
            cnn.add_module("act_fn", nn.Tanh())
        self.layer_init(cnn)
        return cnn

    @staticmethod
    def _conv_output_dim(
        dimension: Sequence[int],
        padding: Sequence[int],
        dilation: Sequence[int],
        kernel_size: Sequence[int],
        stride: Sequence[int],
    ) -> Tuple[int, ...]:
        """Calculates the output height and width based on the input height and
        width to the convolution layer. For parameter definitions see.

        [here](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d).
        # Parameters
        dimension : See above link.
        padding : See above link.
        dilation : See above link.
        kernel_size : See above link.
        stride : See above link.
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    @staticmethod
    def layer_init(cnn) -> None:
        """Initialize layer parameters using Kaiming normal."""

        def _init(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

        if isinstance(cnn, nn.Sequential):
            for layer in cnn:
                _init(layer)
        else:
            _init(cnn)

    def forward(
        self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """input could be either tensor or dict with self.rgb_uuid as key."""
        obs = (
            observations[self.rgb_uuid]
            if isinstance(observations, dict)
            else observations
        )
        return compute_cnn_output(
            self.model,
            self.aug(obs),
            permute_order=None if self.channel_first else (0, 3, 1, 2),
        )


class RandomShiftsAug(nn.Module):
    def __init__(
        self,
        pad: int = 4,
        shift_together_if_cat: bool = False,
        same_shift_if_cat: bool = True,
    ):
        super().__init__()
        self._pad = pad
        self._shift_together_if_cat = shift_together_if_cat
        self._same_shift_if_cat = same_shift_if_cat

    def _forward(
        self, x: torch.Tensor, shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        channel_first = True
        l, b, c, h, w = x.size()
        if c > 10:  # implicit infer, and recently inefficient
            channel_first = False
            x = rearrange(x, "l b h w c -> l b c h w")
            l, b, c, h, w = x.size()
        n = l * b
        x = x.view(n, c, h, w)
        max_dim = max(h, w)
        padding = tuple([self._pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (max_dim + 2 * self._pad)
        x_arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, w + 2 * self._pad, device=x.device, dtype=x.dtype
        )[:w]
        y_arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self._pad, device=x.device, dtype=x.dtype
        )[:h]
        x_arange = x_arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        y_arange = y_arange.unsqueeze(0).repeat(w, 1).unsqueeze(2)
        base_grid = torch.cat([x_arange, y_arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        if shift is None:
            shift = torch.randint(
                0, 2 * self._pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
            )
        shift *= 2.0 / (max_dim + 2 * self._pad)

        grid = base_grid + shift
        grid_sample = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        return (
            rearrange(grid_sample, "(l b) c h w -> l b c h w", l=l, b=b)
            if channel_first
            else rearrange(grid_sample, "(l b) c h w -> l b h w c", l=l, b=b)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l, b, c, h, w = x.size()
        if w == 2 * h and not self._shift_together_if_cat:  # concatenate views case
            img1, img2 = x[..., :h], x[..., h:]
            shift = (
                torch.randint(
                    0,
                    2 * self._pad + 1,
                    size=(l * b, 1, 1, 2),
                    device=x.device,
                    dtype=x.dtype,
                )
                if self._same_shift_if_cat
                else None
            )
            return torch.cat(
                [self._forward(img1, shift=shift), self._forward(img2, shift=shift)],
                dim=-1,
            )
        else:
            return self._forward(x)

    def __repr__(self) -> str:
        return (
            f"{self._get_name()}("
            f"pad={self._pad}, "
            f"shift_together_if_cat={self._shift_together_if_cat}, "
            f"same_shift_if_cat={self._same_shift_if_cat})"
        )


AUG_MAP = dict(random_shift=RandomShiftsAug)
