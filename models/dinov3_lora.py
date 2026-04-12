from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - handled at runtime.
    LoraConfig = None
    get_peft_model = None


DEFAULT_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS


@dataclass
class SemanticForwardState:
    vit_backbone: nn.Module
    tokens: torch.Tensor
    image_size: tuple[int, int]
    hw_tuple: tuple[int, int] | None
    next_block_index: int = 0
    next_level_index: int = 0


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def inject_lora_adapter(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
    target_modules: Iterable[str] = DEFAULT_LORA_TARGETS,
) -> nn.Module:
    """Attach LoRA adapters to a frozen DINOv3 backbone."""

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("PEFT is required to inject LoRA adapters. Install `peft` first.")

    freeze_module(model)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=None,
    )
    lora_model = get_peft_model(model, config)
    return lora_model


def load_dinov3_model(
    repo_or_dir: str,
    model_name: str = "dinov3_vitl16",
    weights: str | None = None,
    source: str = "local",
    **hub_kwargs,
) -> nn.Module:
    """Load a DINOv3 model through torch.hub."""

    load_kwargs = {
        "repo_or_dir": repo_or_dir,
        "model": model_name,
        "source": source,
    }
    if weights is not None:
        load_kwargs["weights"] = weights
    load_kwargs.update(hub_kwargs)
    return torch.hub.load(**load_kwargs)


class DINOv3SemanticBackbone(nn.Module):
    """Wrapper that supports multi-level extraction and staged block-by-block interaction."""

    def __init__(
        self,
        backbone: nn.Module,
        out_indices: Sequence[int] = (5, 11, 17, 23),
        reshape: bool = True,
        return_class_token: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.out_indices = tuple(out_indices)
        self.reshape = reshape
        self.return_class_token = return_class_token

        self.embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", None))
        self.patch_size = self._infer_patch_size(backbone)
        self.num_prefix_tokens = getattr(backbone, "num_prefix_tokens", 1)

    @staticmethod
    def _infer_patch_size(backbone: nn.Module) -> tuple[int, int]:
        patch_embed = getattr(backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            return patch_size
        return (patch_size, patch_size)

    def _to_feature_map(self, feature, image_size: tuple[int, int]) -> torch.Tensor:
        if isinstance(feature, (tuple, list)):
            feature = feature[0]

        if feature.ndim == 4:
            return feature

        if feature.ndim != 3:
            raise ValueError(f"Unsupported DINOv3 feature shape: {tuple(feature.shape)}")

        grid_h = image_size[0] // self.patch_size[0]
        grid_w = image_size[1] // self.patch_size[1]
        patch_tokens = grid_h * grid_w

        if feature.shape[1] != patch_tokens:
            prefix_tokens = feature.shape[1] - patch_tokens
            if prefix_tokens < 0:
                raise ValueError(
                    "Could not reshape DINOv3 tokens into a 2D feature map. "
                    f"Expected at least {patch_tokens} tokens but received {feature.shape[1]}."
                )
            feature = feature[:, prefix_tokens:, :]

        feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[2], grid_h, grid_w)
        return feature

    def _resolve_vit_backbone(self) -> nn.Module:
        backbone = self.backbone

        if hasattr(backbone, "get_base_model"):
            try:
                candidate = backbone.get_base_model()
                if isinstance(candidate, nn.Module):
                    backbone = candidate
            except Exception:
                pass

        for attr_chain in ("base_model.model", "base_model", "model"):
            candidate = backbone
            try:
                for attr in attr_chain.split("."):
                    candidate = getattr(candidate, attr)
                if isinstance(candidate, nn.Module):
                    backbone = candidate
                    break
            except AttributeError:
                continue

        return backbone

    def _build_prefix_tokens(self, vit_backbone: nn.Module, batch_size: int) -> torch.Tensor | None:
        prefix_tokens = []

        cls_token = getattr(vit_backbone, "cls_token", None)
        if cls_token is not None:
            prefix_tokens.append(cls_token.expand(batch_size, -1, -1))

        for register_name in ("reg_token", "register_tokens"):
            register_token = getattr(vit_backbone, register_name, None)
            if register_token is not None:
                prefix_tokens.append(register_token.expand(batch_size, -1, -1))

        if not prefix_tokens:
            return None
        return torch.cat(prefix_tokens, dim=1)

    def _prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        vit_backbone = self._resolve_vit_backbone()

        if hasattr(vit_backbone, "prepare_tokens_with_masks"):
            tokens, hw_tuple = vit_backbone.prepare_tokens_with_masks(x, masks=None)
            return tokens, hw_tuple

        if hasattr(vit_backbone, "prepare_tokens"):
            tokens = vit_backbone.prepare_tokens(x)
            return tokens, None

        if not hasattr(vit_backbone, "patch_embed"):
            raise AttributeError(
                "Could not prepare DINOv3 tokens. The backbone must expose `prepare_tokens_with_masks`, "
                "`prepare_tokens`, or `patch_embed`."
            )

        tokens = vit_backbone.patch_embed(x)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)

        prefix_tokens = self._build_prefix_tokens(vit_backbone, x.shape[0])
        if prefix_tokens is not None:
            tokens = torch.cat([prefix_tokens, tokens], dim=1)

        if hasattr(vit_backbone, "_pos_embed"):
            tokens = vit_backbone._pos_embed(tokens)
        elif hasattr(vit_backbone, "pos_embed") and getattr(vit_backbone, "pos_embed") is not None:
            pos_embed = vit_backbone.pos_embed
            if pos_embed.shape[1] == tokens.shape[1]:
                tokens = tokens + pos_embed

        if hasattr(vit_backbone, "pos_drop"):
            tokens = vit_backbone.pos_drop(tokens)

        if hasattr(vit_backbone, "norm_pre"):
            tokens = vit_backbone.norm_pre(tokens)

        return tokens, None

    def _make_rope(self, vit_backbone: nn.Module, hw_tuple: tuple[int, int] | None):
        if hw_tuple is None or not hasattr(vit_backbone, "rope_embed") or vit_backbone.rope_embed is None:
            return None
        height, width = hw_tuple
        return vit_backbone.rope_embed(H=height, W=width)

    def _norm_output_tokens(self, vit_backbone: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        n_storage_tokens = getattr(vit_backbone, "n_storage_tokens", 0)
        untie_cls_and_patch_norms = getattr(vit_backbone, "untie_cls_and_patch_norms", False)

        if untie_cls_and_patch_norms:
            cls_norm = getattr(vit_backbone, "cls_norm", None)
            patch_norm = getattr(vit_backbone, "norm", None)
            if cls_norm is None or patch_norm is None:
                raise AttributeError(
                    "The DINOv3 backbone declares `untie_cls_and_patch_norms=True` but does not expose "
                    "`cls_norm` and `norm`."
                )
            x_norm_cls_reg = cls_norm(tokens[:, : n_storage_tokens + 1])
            x_norm_patch = patch_norm(tokens[:, n_storage_tokens + 1 :])
            return torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)

        norm = getattr(vit_backbone, "norm", None)
        if norm is None:
            return tokens
        return norm(tokens)

    def _split_prefix_tokens(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> tuple[torch.Tensor | None, torch.Tensor]:
        grid_h = image_size[0] // self.patch_size[0]
        grid_w = image_size[1] // self.patch_size[1]
        patch_tokens = grid_h * grid_w

        prefix_count = tokens.shape[1] - patch_tokens
        if prefix_count < 0:
            raise ValueError(
                "Could not split DINOv3 tokens into prefix and patch tokens. "
                f"Patch token target is {patch_tokens}, but the sequence has {tokens.shape[1]} tokens."
            )

        prefix_tokens = tokens[:, :prefix_count] if prefix_count > 0 else None
        patch_tokens_tensor = tokens[:, prefix_count:, :]
        return prefix_tokens, patch_tokens_tensor

    def _tokens_to_feature_map(self, tokens: torch.Tensor, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        prefix_tokens, patch_tokens = self._split_prefix_tokens(tokens, image_size)
        grid_h = image_size[0] // self.patch_size[0]
        grid_w = image_size[1] // self.patch_size[1]
        feature_map = patch_tokens.transpose(1, 2).reshape(tokens.shape[0], patch_tokens.shape[2], grid_h, grid_w)
        return feature_map, prefix_tokens

    @staticmethod
    def _feature_map_to_tokens(feature_map: torch.Tensor, prefix_tokens: torch.Tensor | None) -> torch.Tensor:
        patch_tokens = feature_map.flatten(2).transpose(1, 2)
        if prefix_tokens is None:
            return patch_tokens
        return torch.cat([prefix_tokens, patch_tokens], dim=1)

    def begin_forward(self, x: torch.Tensor) -> SemanticForwardState:
        """Prepare a staged DINOv3 forward pass that can be paused at interaction points."""

        vit_backbone = self._resolve_vit_backbone()
        if not hasattr(vit_backbone, "blocks"):
            raise AttributeError("The provided DINOv3 backbone does not expose transformer `blocks`.")

        tokens, hw_tuple = self._prepare_tokens(x)
        return SemanticForwardState(
            vit_backbone=vit_backbone,
            tokens=tokens,
            image_size=x.shape[-2:],
            hw_tuple=hw_tuple,
        )

    def _run_blocks_until(self, state: SemanticForwardState, target_block_index: int) -> None:
        if target_block_index < state.next_block_index:
            raise ValueError(
                f"Target block index {target_block_index} has already been passed. "
                f"Next available block index is {state.next_block_index}."
            )

        for block_index in range(state.next_block_index, target_block_index + 1):
            block = state.vit_backbone.blocks[block_index]
            rope = self._make_rope(state.vit_backbone, state.hw_tuple)
            state.tokens = block(state.tokens, rope)
        state.next_block_index = target_block_index + 1

    def forward_to_next_interaction(self, state: SemanticForwardState) -> torch.Tensor:
        """Advance the backbone to the next selected transformer block and expose Z_l."""

        if state.next_level_index >= len(self.out_indices):
            raise ValueError("All configured DINOv3 interaction levels have already been consumed.")

        target_block_index = self.out_indices[state.next_level_index]
        self._run_blocks_until(state, target_block_index)
        feature_map, _ = self._tokens_to_feature_map(state.tokens, state.image_size)
        return feature_map

    def apply_interaction_result(self, state: SemanticForwardState, refined_feature_map: torch.Tensor) -> torch.Tensor:
        """Replace the current patch tokens with Z_hat_l and return the normalized decoder feature."""

        _, prefix_tokens = self._tokens_to_feature_map(state.tokens, state.image_size)
        state.tokens = self._feature_map_to_tokens(refined_feature_map, prefix_tokens)

        tokens_for_output = self._norm_output_tokens(state.vit_backbone, state.tokens)
        output_feature_map, _ = self._tokens_to_feature_map(tokens_for_output, state.image_size)
        state.next_level_index += 1
        return output_feature_map

    def _forward_with_interactions(
        self,
        x: torch.Tensor,
        structural_features: Sequence[torch.Tensor] | None = None,
        interaction_blocks: Sequence[nn.Module] | None = None,
    ) -> list[torch.Tensor]:
        state = self.begin_forward(x)
        outputs: list[torch.Tensor | None] = [None] * len(self.out_indices)
        for level in range(len(self.out_indices)):
            feature_map = self.forward_to_next_interaction(state)

            if structural_features is not None and interaction_blocks is not None:
                feature_map = interaction_blocks[level](feature_map, structural_features[level])

            outputs[level] = self.apply_interaction_result(state, feature_map)

        if any(output is None for output in outputs):
            raise ValueError(
                f"Requested {len(self.out_indices)} semantic outputs from DINOv3, but collected "
                f"{sum(output is not None for output in outputs)}. "
                "Please verify that the selected `out_indices` exist in the backbone."
            )

        return [output for output in outputs if output is not None]

    def forward(
        self,
        x: torch.Tensor,
        structural_features: Sequence[torch.Tensor] | None = None,
        interaction_blocks: Sequence[nn.Module] | None = None,
    ) -> list[torch.Tensor]:
        if (structural_features is None) ^ (interaction_blocks is None):
            raise ValueError("`structural_features` and `interaction_blocks` must be provided together.")

        if structural_features is not None and interaction_blocks is not None:
            if len(structural_features) != len(self.out_indices):
                raise ValueError("The number of structural features must match the number of DINOv3 output indices.")
            if len(interaction_blocks) != len(self.out_indices):
                raise ValueError("The number of CSSI blocks must match the number of DINOv3 output indices.")
            return self._forward_with_interactions(
                x,
                structural_features=structural_features,
                interaction_blocks=interaction_blocks,
            )

        if not hasattr(self.backbone, "get_intermediate_layers"):
            return self._forward_with_interactions(x)

        raw_features = self.backbone.get_intermediate_layers(
            x,
            n=list(self.out_indices),
            reshape=self.reshape,
            return_class_token=self.return_class_token,
        )
        return [self._to_feature_map(feature, x.shape[-2:]) for feature in raw_features]
