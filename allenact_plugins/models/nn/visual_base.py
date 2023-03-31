from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from allenact.base_abstractions.sensor import SpaceDict


class VisualBase(nn.Module):
    """Base class of visual backbone followed by a fc layer for encoding rgb
    obs. Takes in observations (of type gym.spaces.dict) and produces an
    embedding of the `rgb_uuid` component.

    # Attributes
        observation_space: The observation_space of the agent, should have `rgb_uuid` component.
        output_size : The size of the embedding vector to produce.
        rgb_uuid: key in observation space for visual observation
        num_stacked_frames: num of frames that stacked in the color channel. None for no frame-stacking
        channel_first: whether chw dim for rgb obs
        flatten: whether flatten for final embedding
        output_relu: whether include relu for the final embedding
    """

    def __init__(
        self,
        observation_space: SpaceDict,
        *,
        output_size: int,
        rgb_uuid: str,
        num_stacked_frames: Optional[int] = None,
        channel_first: bool = True,
        # flatten: bool = True,
        # output_relu: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.rgb_uuid = rgb_uuid
        self.channel_first = channel_first
        assert (
            self.rgb_uuid in observation_space.spaces
        ), f"{self.rgb_uuid} not in {observation_space.spaces}"
        self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[
            0 if channel_first else 2
        ] * (num_stacked_frames or 1)
        assert self._n_input_rgb >= 0
        input_rgb_dims = (
            observation_space.spaces[self.rgb_uuid].shape[1:]
            if channel_first
            else observation_space.spaces[self.rgb_uuid].shape[:2]
        )
        self.output_size = output_size
        self.model = self.setup_model(
            output_size=output_size,
            input_dims=np.array(input_rgb_dims, dtype=np.float32),
            input_channels=self._n_input_rgb,
            # flatten=flatten,
            # output_relu=output_relu,
            **kwargs,
        )

    def setup_model(
        self,
        output_size: int,
        input_dims: np.ndarray,
        input_channels: int,
        flatten: bool = True,
        output_relu: bool = True,
        **kwargs,
    ) -> nn.Module:
        """function for initialize a visual encoder."""
        raise NotImplementedError

    def forward(
        self, observations: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        raise NotImplementedError
