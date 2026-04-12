from __future__ import annotations

import argparse

import torch

from models import LoRASettings, build_lasa_net_from_loaded_backbone


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaSA-Net forward test with DINOv3 + LoRA.")
    parser.add_argument("--repo-or-dir", default="dinov3-main", help="Local path to the DINOv3 repository.")
    parser.add_argument(
        "--weights",
        default="DINOv3_Pretrain/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        help="Path to the local DINOv3 checkpoint.",
    )
    parser.add_argument("--model-name", default="dinov3_vitl16", help="DINOv3 model name.")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--image-size", type=int, default=512, help="Input image size for the forward test.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for the forward test.")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")
    parser.add_argument(
        "--cnn-model-name",
        default="mobilenetv3_small_100",
        help="CNN model used in the structural prior stream.",
    )
    parser.add_argument(
        "--disable-cnn-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights for the CNN prior stream.",
    )
    return parser.parse_args()


def run_forward_test(args: argparse.Namespace) -> None:
    torch.manual_seed(0)

    dinov3 = torch.hub.load(
        repo_or_dir=args.repo_or_dir,
        model=args.model_name,
        source="local",
        weights=args.weights,
    )

    model = build_lasa_net_from_loaded_backbone(
        dinov3_backbone=dinov3,
        num_classes=args.num_classes,
        lora_settings=LoRASettings(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        ),
        cnn_model_name=args.cnn_model_name,
        cnn_pretrained=not args.disable_cnn_pretrained,
    )
    model.eval()

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")

    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Input shape:  {tuple(inputs.shape)}")
    print(f"Output shape: {tuple(outputs.shape)}")

    expected_shape = (args.batch_size, args.num_classes, args.image_size, args.image_size)
    assert outputs.shape == expected_shape, (
        f"Unexpected output shape {tuple(outputs.shape)}; expected {expected_shape}."
    )
    print("Forward test passed.")


if __name__ == "__main__":
    run_forward_test(parse_args())
