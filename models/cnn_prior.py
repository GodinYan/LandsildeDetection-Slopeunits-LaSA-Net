from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover - handled at runtime.
    timm = None


class StructuralPriorStream(nn.Module):
    """CNN branch that provides multi-scale structural priors."""

    def __init__(
        self,
        model_name: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("`timm` is required to build the structural prior stream.")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        self.out_channels = list(self.backbone.feature_info.channels())
        self.reductions = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.backbone(x))
