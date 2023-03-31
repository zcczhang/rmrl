from typing import Sequence, Union

import attr
import numpy as np

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor


@attr.s(kw_only=True)
class ClipResNetPreprocessorMixin:
    sensors: Sequence[Sensor] = attr.ib()
    rgb_output_uuid: str = attr.ib()
    depth_output_uuid: str = attr.ib(default=None)
    clip_model_type: str = attr.ib()
    pool: bool = attr.ib(default=False)

    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []

        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_means)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_sds)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
            )
            < 1e-5
        )

        if rgb_sensor is not None:
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=rgb_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid=self.rgb_output_uuid,
                    input_img_height_width=(rgb_sensor.height, rgb_sensor.width),
                )
            )

        depth_sensor = next(
            (s for s in self.sensors if isinstance(s, DepthSensor)), None
        )
        if depth_sensor is not None:
            assert self.depth_output_uuid is not None
            preprocessors.append(
                ClipResNetPreprocessor(
                    rgb_input_uuid=depth_sensor.uuid,
                    clip_model_type=self.clip_model_type,
                    pool=self.pool,
                    output_uuid=self.depth_output_uuid,
                    input_img_height_width=(depth_sensor.height, depth_sensor.width),
                )
            )

        return preprocessors
