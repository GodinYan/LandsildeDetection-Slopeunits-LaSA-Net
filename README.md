# LandsildeDetection-Slopeunits-LaSA-Net
# LaSA-Net

## Installation

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Forward test with local DINOv3 + LoRA

This test script follows the same usage pattern as the training code: it first loads DINOv3 with `torch.hub.load(...)`, injects LoRA, builds LaSA-Net, and then runs a forward pass.

```bash
python test_forward.py
```

You can also explicitly specify the local DINOv3 repository path and checkpoint path:

```bash
python test_forward.py --repo-or-dir dinov3-main --weights DINOv3_Pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

### 2. Build the model in training code

```python
import torch

from models import LoRASettings, build_lasa_net_from_loaded_backbone

dinov3 = torch.hub.load(
    repo_or_dir="dinov3-main",
    model="dinov3_vitl16",
    source="local",
    weights="DINOv3_Pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
)

model = build_lasa_net_from_loaded_backbone(
    dinov3_backbone=dinov3,
    num_classes=2,
    intermediate_indices=(5, 11, 17, 23),
    lora_settings=LoRASettings(rank=16, alpha=16, dropout=0.0),
)
```

