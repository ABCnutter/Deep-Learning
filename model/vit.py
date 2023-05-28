import torch
import torch.nn as nn
from functools import partial
from typing import Callable
from math import sqrt
from torch import Tensor
from torchvision.models.vision_transformer import (
    Encoder,
    VisionTransformer,
    ViT_H_14_Weights,
    model_urls,
)


model_urls["vit_h_14"] = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1.url


class Encoder(Encoder):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # del self.ln

    def forward(self, x: Tensor):
        print(x.shape)
        torch._assert(
            x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}"
        )
        n, l, _ = x.shape
        features = []
        x = x + self.pos_embedding

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            y = self.ln(x)
            y = y[:, 1:, :]
            y = y.permute(0, 1, 2)
            y = y.reshape(n, -1, int(sqrt(l)), int(sqrt(l)))
            features.append(y)
        return features


class VisionTransformerEncoder(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            **kwargs,
        )
        seq_length = (image_size // patch_size) ** 2 + 1

        encoder_kwargs = dict(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
        )
        self.encoder = Encoder(**encoder_kwargs)

        del self.heads

    def forward(self, x: Tensor):
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        return x

    def load_state_dict(
        self, weights_name: str, progress: bool = True, strict: bool = True
    ):
        if weights_name is None:
            state_dict = None
            print("No pretrained weights exist for this model.")
        elif isinstance(weights_name, str):
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls[weights_name], progress=progress
            )
        if state_dict is not None and not set(list(self.state_dict().keys())).issubset(
            list(set(state_dict.keys()))
        ):
            state_dict = self._renew_state_dict(
                state_dict, list(self.state_dict().keys())
            )

            return super().load_state_dict(state_dict, strict)

    def _renew_state_dict(
        self,
        state_dict,
        model_state_dict_keys,
    ):
        for k in list(state_dict.keys()):
            if "mlp.linear_1" in k:
                state_dict[k.replace("linear_1", "0")] = state_dict.pop(k)

            elif "mlp.linear_2" in k:
                state_dict[k.replace("linear_2", "3")] = state_dict.pop(k)
        new_state_dict = {}
        for k in model_state_dict_keys:
            assert k in state_dict.keys(), f"{k} not in torch_state_dict"
            new_state_dict[k] = state_dict[k]

        return new_state_dict


def _build_vit(
    image_size: int = 224,
    patch_size: int = 16,
    num_layers: int = 12,
    num_heads: int = 12,
    hidden_dim: int = 768,
    mlp_dim: int = 3072,
    weights_name: str = None,
    progress: bool = True,
    **kwargs,
):
    model = VisionTransformerEncoder(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights_name is not None:
        assert weights_name in list(
            model_urls.keys()
        ), f"{weights_name} not in {list(model_urls.keys())}"
        model.load_state_dict(weights_name, progress)
    return model


def vit_b_16(
    num_layers=12,
    num_heads=12,
    weights_name: str = "vit_b_16",
    progress: bool = True,
    **kwargs,
):
    if weights_name is not None:
        assert (
            weights_name == "vit_b_16"
        ), f"{weights_name} not in {model_urls['vit_b_16']}"

    return _build_vit(
        image_size=224,
        patch_size=16,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=768,
        mlp_dim=3072,
        weights_name=weights_name,
        progress=progress,
        **kwargs,
    )


def vit_b_32(
    num_layers=12,
    num_heads=12,
    weights_name: str = "vit_b_32",
    progress: bool = True,
    **kwargs,
):
    if weights_name is not None:
        assert (
            weights_name == "vit_b_32"
        ), f"{weights_name} not in {model_urls['vit_b_32']}"

    return _build_vit(
        image_size=224,
        patch_size=32,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=786,
        mlp_dim=3072,
        weights_name=weights_name,
        progress=progress,
        **kwargs,
    )


def vit_l_16(
    num_layers=24,
    num_heads=16,
    weights_name: str = "vit_l_16",
    progress: bool = True,
    **kwargs,
):
    if weights_name is not None:
        assert (
            weights_name == "vit_l_16"
        ), f"{weights_name} not in {model_urls['vit_l_16']}"

    return _build_vit(
        image_size=224,
        patch_size=16,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=1024,
        mlp_dim=4096,
        weights_name=weights_name,
        progress=progress,
        **kwargs,
    )


def vit_l_32(
    num_layers=24,
    num_heads=16,
    weights_name: str = "vit_l_32",
    progress: bool = True,
    **kwargs,
):
    if weights_name is not None:
        assert (
            weights_name == "vit_l_32"
        ), f"{weights_name} not in {model_urls['vit_l_32']}"

    return _build_vit(
        image_size=224,
        patch_size=32,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=1024,
        mlp_dim=4096,
        weights_name=weights_name,
        progress=progress,
        **kwargs,
    )


def vit_h_14(
    num_layers=32,
    num_heads=16,
    weights_name: str = "vit_h_14",
    progress: bool = True,
    **kwargs,
):
    if weights_name is not None:
        assert (
            weights_name == "vit_h_14"
        ), f"{weights_name} not in {model_urls['vit_h_14']}"

    return _build_vit(
        image_size=224,
        patch_size=14,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=1280,
        mlp_dim=5120,
        weights_name=weights_name,
        progress=progress,
        **kwargs,
    )


if __name__ == "__main__":
    model = vit_b_16(num_layers=4, num_heads=4, weights_name="vit_b_16", progress=True)
    inputs = torch.rand(size=(2, 3, 224, 224), dtype=torch.float32)
    outputs = model(inputs)
    print([out.shape for out in outputs])
