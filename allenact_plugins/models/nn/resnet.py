from typing import Optional, Dict, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from allenact.base_abstractions.sensor import SpaceDict
from allenact_plugins.models.nn.visual_base import VisualBase


class ResNetEncoder(VisualBase):
    def __init__(
        self,
        observation_space: SpaceDict,
        *,
        output_size: int,
        rgb_uuid: Optional[str],
        num_stacked_frames: Optional[int] = None,
        channel_first: bool = True,
        resnet_name: str = "resnet18",
        pretrained: bool = False,
        freeze: bool = False,
        **kwargs,
    ):
        assert (
            resnet_name in models.resnet.__all__
        ), f"{resnet_name} not in {models.resnet.__all__}"
        self._pretrained = pretrained
        self._freeze = freeze
        super().__init__(
            observation_space=observation_space,
            output_size=output_size,
            rgb_uuid=rgb_uuid,
            num_stacked_frames=num_stacked_frames,
            channel_first=channel_first,
            resnet_name=resnet_name,
            **kwargs,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_planes, output_size)

    def setup_model(
        self,
        output_size: int,
        input_dims: np.ndarray,
        input_channels: int,
        resnet_name: Literal["resnet18", "resnet18_gn"] = "resnet18",
        **kwargs,
    ) -> nn.Module:
        resnet = getattr(models.resnet, resnet_name)(pretrained=self._pretrained)
        if input_channels != 3:  # if frame-stacking
            resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.in_planes = resnet.inplanes
        resnet = nn.Sequential(*(list(resnet.children())[:-2]))

        if self._freeze:
            for param in resnet.parameters():
                param.requires_grad = False
            resnet.eval()

        return resnet

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, dict):
            x = x[self.rgb_uuid]
        leading_dim = x.shape[:-3]

        nagents: Optional[int] = x.shape[3] if len(x.shape) == 6 else None
        # Make FLAT_BATCH = nsteps * nsamplers (* nagents)
        x = x.view((-1,) + x.shape[2 + int(nagents is not None) :])

        if self._freeze:
            with torch.no_grad():
                x = self.model(x)
        else:
            x = self.model(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(*leading_dim, self.output_size)
        return x
