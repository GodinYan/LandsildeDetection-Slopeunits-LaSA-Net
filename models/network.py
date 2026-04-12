from __future__ import annotations

import torch
import torch.nn as nn

from .cnn_prior import StructuralPriorStream
from .cssi import CSSIBlock
from .decoder import HRMSDDecoder
from .dinov3_lora import (
    DINOv3SemanticBackbone,
    LoRASettings,
    freeze_module,
    inject_lora_adapter,
    load_dinov3_model,
)


class LaSANet(nn.Module):
    """Landslide-Aware Scale-Adaptive Network."""

    def __init__(
        self,
        semantic_backbone: DINOv3SemanticBackbone,
        structural_prior: StructuralPriorStream,
        semantic_dim: int,
        structural_dims: list[int],
        cssi_dim: int = 256,
        cssi_heads: int = 8,
        kv_downsample_strides: tuple[int, int, int, int] = (4, 2, 1, 1),
        decoder_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if len(structural_dims) != 4:
            raise ValueError("LaSANet expects four structural pyramid channels.")
        if len(kv_downsample_strides) != 4:
            raise ValueError("LaSANet expects four CSSI downsample strides.")

        self.semantic_backbone = semantic_backbone
        self.structural_prior = structural_prior

        self.cssi_blocks = nn.ModuleList(
            [
                CSSIBlock(
                    semantic_dim=semantic_dim,
                    structural_dim=structural_dims[index],
                    attn_dim=cssi_dim,
                    num_heads=cssi_heads,
                    kv_downsample=kv_downsample_strides[index],
                )
                for index in range(4)
            ]
        )
        self.decoder = HRMSDDecoder(
            semantic_dim=semantic_dim,
            structural_dims=structural_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m1, m2, m3, m4 = self.structural_prior(x)

        semantic_state = self.semantic_backbone.begin_forward(x)

        z_1 = self.semantic_backbone.forward_to_next_interaction(semantic_state)
        z_hat_1 = self.cssi_blocks[0](z_1, m1)
        z_hat_1 = self.semantic_backbone.apply_interaction_result(semantic_state, z_hat_1)

        z_2 = self.semantic_backbone.forward_to_next_interaction(semantic_state)
        z_hat_2 = self.cssi_blocks[1](z_2, m2)
        z_hat_2 = self.semantic_backbone.apply_interaction_result(semantic_state, z_hat_2)

        z_3 = self.semantic_backbone.forward_to_next_interaction(semantic_state)
        z_hat_3 = self.cssi_blocks[2](z_3, m3)
        z_hat_3 = self.semantic_backbone.apply_interaction_result(semantic_state, z_hat_3)

        z_4 = self.semantic_backbone.forward_to_next_interaction(semantic_state)
        z_hat_4 = self.cssi_blocks[3](z_4, m4)
        z_hat_4 = self.semantic_backbone.apply_interaction_result(semantic_state, z_hat_4)

        # Stage 3: HRMSD fuses the refined semantic pyramid {Z_hat_1, ..., Z_hat_4}
        # with the structural pyramid {M1, ..., M4} and produces dense segmentation logits.
        segmentation_logits = self.decoder(
            semantic_features=[z_hat_1, z_hat_2, z_hat_3, z_hat_4],
            structural_features=[m1, m2, m3, m4],
            output_size=x.shape[-2:],
        )
        return segmentation_logits


def build_lasa_net_from_loaded_backbone(
    dinov3_backbone: nn.Module,
    num_classes: int = 2,
    intermediate_indices: tuple[int, int, int, int] = (5, 11, 17, 23),
    lora_settings: LoRASettings | None = LoRASettings(),
    cnn_model_name: str = "mobilenetv3_small_100",
    cnn_pretrained: bool = True,
    cnn_out_indices: tuple[int, int, int, int] = (1, 2, 3, 4),
    cssi_dim: int = 256,
    cssi_heads: int = 8,
    kv_downsample_strides: tuple[int, int, int, int] = (4, 2, 1, 1),
    decoder_dim: int = 256,
) -> LaSANet:
    """Build LaSA-Net from an already loaded DINOv3 backbone."""

    backbone = dinov3_backbone
    if lora_settings is not None:
        backbone = inject_lora_adapter(
            backbone,
            rank=lora_settings.rank,
            alpha=lora_settings.alpha,
            dropout=lora_settings.dropout,
            target_modules=lora_settings.target_modules,
        )
    else:
        freeze_module(backbone)

    semantic_backbone = DINOv3SemanticBackbone(
        backbone=backbone,
        out_indices=intermediate_indices,
        reshape=True,
        return_class_token=False,
    )
    structural_prior = StructuralPriorStream(
        model_name=cnn_model_name,
        pretrained=cnn_pretrained,
        out_indices=cnn_out_indices,
    )

    semantic_dim = semantic_backbone.embed_dim
    if semantic_dim is None:
        raise ValueError("Could not infer the DINOv3 embedding dimension from the provided backbone.")

    return LaSANet(
        semantic_backbone=semantic_backbone,
        structural_prior=structural_prior,
        semantic_dim=semantic_dim,
        structural_dims=structural_prior.out_channels,
        cssi_dim=cssi_dim,
        cssi_heads=cssi_heads,
        kv_downsample_strides=kv_downsample_strides,
        decoder_dim=decoder_dim,
        num_classes=num_classes,
    )


def build_lasa_net(
    dinov3_repo_or_dir: str,
    dinov3_weights: str | None = None,
    dinov3_model_name: str = "dinov3_vitl16",
    num_classes: int = 2,
    intermediate_indices: tuple[int, int, int, int] = (5, 11, 17, 23),
    lora_settings: LoRASettings | None = LoRASettings(),
    cnn_model_name: str = "mobilenetv3_small_100",
    cnn_pretrained: bool = True,
    cnn_out_indices: tuple[int, int, int, int] = (1, 2, 3, 4),
    cssi_dim: int = 256,
    cssi_heads: int = 8,
    kv_downsample_strides: tuple[int, int, int, int] = (4, 2, 1, 1),
    decoder_dim: int = 256,
    source: str = "local",
    **hub_kwargs,
) -> LaSANet:
    """Factory that builds the complete LaSA-Net from a DINOv3 checkpoint."""

    dinov3_backbone = load_dinov3_model(
        repo_or_dir=dinov3_repo_or_dir,
        model_name=dinov3_model_name,
        weights=dinov3_weights,
        source=source,
        **hub_kwargs,
    )

    return build_lasa_net_from_loaded_backbone(
        dinov3_backbone=dinov3_backbone,
        num_classes=num_classes,
        intermediate_indices=intermediate_indices,
        lora_settings=lora_settings,
        cnn_model_name=cnn_model_name,
        cnn_pretrained=cnn_pretrained,
        cnn_out_indices=cnn_out_indices,
        cssi_dim=cssi_dim,
        cssi_heads=cssi_heads,
        kv_downsample_strides=kv_downsample_strides,
        decoder_dim=decoder_dim,
    )
