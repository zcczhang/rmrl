from typing import List, Optional, Any, cast, Dict, Tuple, Union

import clip
import gym
import numpy as np
import torch
import torch.nn as nn
from clip.model import CLIP

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.misc_utils import prepare_locals_for_super

__all__ = ["ClipResNetPreprocessor"]


class ClipResNetEmbedder(nn.Module):
    def __init__(
        self,
        model: CLIP,
        *,
        pool: bool = False,
        pooling_type: str = "avg",
        pass_then_cat_features: bool = True,
    ):
        super().__init__()
        self.model = model
        self.pool = pool
        self.pooling_type = pooling_type
        self.pass_then_cat_features = pass_then_cat_features

        if not pool:
            self.model.visual.attnpool = nn.Identity()
        elif self.pooling_type == "attn":
            pass
        elif self.pooling_type == "avg":
            self.model.visual.attnpool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=-3, end_dim=-1)
            )
        else:
            raise NotImplementedError("`pooling_type` must be 'avg' or 'attn'.")

        self.eval()

    def forward(
        self,
        img: Union[np.ndarray, torch.Tensor],
        text: Optional[Union[str, List[str], np.ndarray, torch.Tensor]] = None,
        *,
        normalize_img_embed: bool = False,
        normalize_text_embed: bool = False,
    ):
        with torch.no_grad():
            b, c, h, w = img.shape
            assert c == 3, img.shape
            # L(=1), D, 7, 7  for 224x244, ..., 7, 14 for 224x448
            visual_embed = self.model.visual(img)

            if normalize_img_embed and self.pool:
                visual_embed = visual_embed / visual_embed.norm(
                    p=2, dim=-1, keepdim=True
                )

            if text is None:
                return visual_embed, None
            if isinstance(text, (str, list, np.ndarray)):
                # L (=1), 77, same context padding as clip (too long?)
                text = clip.tokenize(text)
            assert isinstance(text, torch.Tensor)
            # L, 1024
            text_embed = self.model.encode_text(text)
            if normalize_text_embed:
                text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
            return visual_embed, text_embed


class ClipResNetPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        *,
        rgb_input_uuid: str,
        prompt_input_uuid: Optional[str] = None,
        clip_model_type: str,
        pool: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        input_img_height_width: Tuple[int, int] = (224, 224),
        output_uuid: str,
        normalize_img_features: bool = False,
        normalize_text_features: bool = False,
        **kwargs: Any,
    ):
        assert clip_model_type in clip.available_models()
        assert pool is False or input_img_height_width == (224, 224)
        assert all(iis % 32 == 0 for iis in input_img_height_width)

        output_height_width = tuple(iis // 32 for iis in input_img_height_width)
        if clip_model_type == "RN50":
            output_shape = (2048,) + output_height_width
        elif clip_model_type == "RN50x16":
            output_shape = (3072,) + output_height_width
        else:
            raise NotImplementedError(
                f"Currently `clip_model_type` must be one of 'RN50' or 'RN50x16' for ResNet Backbone"
            )

        if pool:
            output_shape = output_shape[:1]

        self.clip_model_type = clip_model_type

        self.pool = pool

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._model: Optional[ClipResNetEmbedder] = None

        low = -1 if normalize_img_features else -np.inf
        high = 1 if normalize_img_features else np.inf
        shape = output_shape

        input_uuids = [rgb_input_uuid]
        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)
        if prompt_input_uuid is not None:
            input_uuids.append(prompt_input_uuid)  # assume idx after rgb
            clip_text_embedding_dim = (1024,)
            observation_space = gym.spaces.Dict(
                {
                    rgb_input_uuid: observation_space,
                    # `normalize_text_embed=True` by default
                    prompt_input_uuid: gym.spaces.Box(
                        low=-1 if normalize_text_features else -np.inf,
                        high=1 if normalize_text_features else np.inf,
                        shape=clip_text_embedding_dim,
                    ),
                }
            )
        self.normalize_img_features = normalize_img_features
        self.normalize_text_features = normalize_text_features
        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def model(self) -> ClipResNetEmbedder:
        if self._model is None:
            self._model = ClipResNetEmbedder(
                clip.load(self.clip_model_type, device=self.device)[0], pool=self.pool
            ).to(self.device)
            for module in self._model.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self._model.eval()
            # debug_model_info(self._model, trainable=False)
        return self._model

    def to(self, device: torch.device) -> "ClipResNetPreprocessor":
        self._model = self.model.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        has_prompt = len(self.input_uuids) > 1
        img = (
            obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        )  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if img.shape[1] == 1:
            img = img.repeat(1, 3, 1, 1)
        # assume idx after rgb
        prompt = obs[self.input_uuids[1]].to(self.device) if has_prompt else None
        img_embed, prompt_embed = self.model(
            img,
            prompt,
            normalize_img_embed=self.normalize_img_features,
            normalize_text_embed=self.normalize_text_features,
        )
        if not has_prompt:
            return img_embed.float()
        img_embed, prompt_embed = img_embed.float(), prompt_embed.float()
        return {self.input_uuids[0]: img_embed, self.input_uuids[1]: prompt_embed}


class ClipViTEmbedder(nn.Module):
    def __init__(self, model: CLIP, class_emb_only: bool = False):
        super().__init__()
        self.model = model
        self.model.visual.transformer.resblocks = nn.Sequential(
            *list(self.model.visual.transformer.resblocks)[:-1]
        )
        self.class_emb_only = class_emb_only

        self.eval()

    def forward(self, x):
        m = self.model.visual
        with torch.no_grad():
            x = m.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    m.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + m.positional_embedding.to(x.dtype)
            x = m.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = m.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            if self.class_emb_only:
                return x[:, 0, :]
            else:
                return x


class ClipViTPreprocessor(Preprocessor):
    """Preprocess RGB or depth image using a ResNet model with CLIP model
    weights."""

    CLIP_RGB_MEANS = (0.48145466, 0.4578275, 0.40821073)
    CLIP_RGB_STDS = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        rgb_input_uuid: str,
        clip_model_type: str,
        class_emb_only: bool,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[torch.device]] = None,
        **kwargs: Any,
    ):
        assert clip_model_type in clip.available_models()

        if clip_model_type == "ViT-B/32":
            output_shape = (7 * 7 + 1, 768)
        elif clip_model_type == "ViT-B/16":
            output_shape = (14 * 14 + 1, 768)
        elif clip_model_type == "ViT-L/14":
            output_shape = (16 * 16 + 1, 1024)
        else:
            raise NotImplementedError(
                f"Currently `clip_model_type` must be one of 'ViT-B/32', 'ViT-B/16', or 'ViT-B/14'"
            )

        if class_emb_only:
            output_shape = output_shape[1:]

        self.clip_model_type = clip_model_type

        self.class_emb_only = class_emb_only

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or cast(
            List[torch.device], list(range(torch.cuda.device_count()))
        )
        self._vit: Optional[ClipViTEmbedder] = None

        low = -np.inf
        high = np.inf
        shape = output_shape

        input_uuids = [rgb_input_uuid]
        assert (
            len(input_uuids) == 1
        ), "resnet preprocessor can only consume one observation type"

        observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        super().__init__(**prepare_locals_for_super(locals()))

    @property
    def vit(self) -> ClipViTEmbedder:
        if self._vit is None:
            self._vit = ClipViTEmbedder(
                model=clip.load(self.clip_model_type, device=self.device)[0],
                class_emb_only=self.class_emb_only,
            ).to(self.device)
            for module in self._vit.modules():
                if "BatchNorm" in type(module).__name__:
                    module.momentum = 0.0
            self._vit.eval()
        return self._vit

    def to(self, device: torch.device) -> "ClipViTPreprocessor":
        self._vit = self.vit.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        # If the input is depth, repeat it across all 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.vit(x).float()
        return x
