#!/usr/bin/env python3
"""Small MONAI segmentation demo with optional distributed training.

This script is intentionally portfolio-friendly:
- synthetic 3D data, so it runs without external datasets
- MONAI transforms, UNet, Dice loss, and sliding-window inference
- optional PyTorch DDP support for local multi-GPU or SLURM launches
- JSON metrics output for reproducibility and easy reporting
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset, DistributedSampler, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)
from monai.utils import set_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MONAI 3D segmentation portfolio demo")
    parser.add_argument("--dataset", choices=["synthetic", "msd_spleen"], default="synthetic")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/synthetic_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/run"))
    parser.add_argument("--num-samples", type=int, default=24, help="number of synthetic image/label pairs")
    parser.add_argument("--cache-rate", type=float, default=0.0, help="cache rate for DecathlonDataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics, and artifacts to MLflow")
    parser.add_argument("--mlflow-experiment", type=str, default="monai-portfolio-demo")
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="initialize torch distributed from environment variables set by torchrun/srun",
    )
    return parser.parse_args()


def in_distributed_mode(args: argparse.Namespace) -> bool:
    return args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    is_distributed = in_distributed_mode(args)
    if not is_distributed:
        return False, 0, 0, 1

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = dist.get_world_size()
    return True, rank, local_rank, world_size


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, rank: int) -> None:
    set_determinism(seed=seed)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def is_main_process(rank: int) -> bool:
    return rank == 0


def maybe_generate_synthetic_dataset(data_dir: Path, num_samples: int, image_size: int, rank: int) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    existing_images = sorted(data_dir.glob("img*.nii.gz"))
    existing_labels = sorted(data_dir.glob("seg*.nii.gz"))
    if len(existing_images) >= num_samples and len(existing_labels) >= num_samples:
        return

    if not is_main_process(rank):
        return

    rng = np.random.default_rng(seed=0)
    for index in range(num_samples):
        image_path = data_dir / f"img{index}.nii.gz"
        label_path = data_dir / f"seg{index}.nii.gz"
        if image_path.exists() and label_path.exists():
            continue
        image, label = create_test_image_3d(
            image_size,
            image_size,
            image_size,
            num_objs=int(rng.integers(4, 8)),
            num_seg_classes=1,
            channel_dim=-1,
            noise_max=0.2,
        )
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)


def sync_if_needed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()


def build_datasets(data_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    images = sorted(data_dir.glob("img*.nii.gz"))
    labels = sorted(data_dir.glob("seg*.nii.gz"))
    records = [{"img": str(image), "seg": str(label)} for image, label in zip(images, labels)]
    split_idx = max(1, int(len(records) * 0.75))
    train_files = records[:split_idx]
    val_files = records[split_idx:]
    return train_files, val_files


def build_synthetic_transforms(roi_size: int) -> tuple[Compose, Compose]:
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys=["img"]),
            RandCropByPosNegLabeld(
                keys=["img", "seg"],
                label_key="seg",
                spatial_size=(roi_size, roi_size, roi_size),
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=(0, 2)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            ScaleIntensityd(keys=["img"]),
        ]
    )
    return train_transforms, val_transforms


def build_msd_spleen_transforms(roi_size: int) -> tuple[Compose, Compose]:
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(roi_size, roi_size, roi_size),
                pos=1,
                neg=1,
                num_samples=2,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )
    return train_transforms, val_transforms


def maybe_prepare_msd_spleen_dataset(data_dir: Path, rank: int, seed: int, cache_rate: float, num_workers: int) -> None:
    if not is_main_process(rank):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    DecathlonDataset(
        root_dir=str(data_dir),
        task="Task09_Spleen",
        section="training",
        transform=(),
        download=True,
        seed=seed,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )


def build_dataset_objects(
    args: argparse.Namespace,
    rank: int,
    is_distributed: bool,
) -> tuple[Dataset, Dataset, tuple[str, str], int]:
    if args.dataset == "synthetic":
        maybe_generate_synthetic_dataset(args.data_dir, args.num_samples, args.image_size, rank)
        sync_if_needed(is_distributed)
        train_files, val_files = build_datasets(args.data_dir)
        train_transforms, val_transforms = build_synthetic_transforms(args.roi_size)
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
        return train_ds, val_ds, ("img", "seg"), len(train_files) + len(val_files)

    maybe_prepare_msd_spleen_dataset(
        data_dir=args.data_dir,
        rank=rank,
        seed=args.seed,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    sync_if_needed(is_distributed)
    train_transforms, val_transforms = build_msd_spleen_transforms(args.roi_size)
    train_ds = DecathlonDataset(
        root_dir=str(args.data_dir),
        task="Task09_Spleen",
        section="training",
        transform=train_transforms,
        download=False,
        seed=args.seed,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    val_ds = DecathlonDataset(
        root_dir=str(args.data_dir),
        task="Task09_Spleen",
        section="validation",
        transform=val_transforms,
        download=False,
        seed=args.seed,
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    return train_ds, val_ds, ("image", "label"), len(train_ds) + len(val_ds)


def build_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(train_ds, even_divisible=True, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, sampler


def build_model(device: torch.device, is_distributed: bool, local_rank: int) -> torch.nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )
    return model


def setup_mlflow(args: argparse.Namespace, rank: int):
    if not args.mlflow or not is_main_process(rank):
        return None, nullcontext()

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow logging was requested, but mlflow is not installed. "
            "Run `python3 -m pip install mlflow` or reinstall from requirements.txt."
        ) from exc

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_context = mlflow.start_run(run_name=args.mlflow_run_name)
    return mlflow, run_context


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    roi_size: int,
    sw_batch_size: int,
    data_keys: tuple[str, str],
) -> float:
    metric = DiceMetric(include_background=True, reduction="mean")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    image_key, label_key = data_keys

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch[image_key].to(device)
            labels = batch[label_key].to(device)
            outputs = sliding_window_inference(
                inputs=images,
                roi_size=(roi_size, roi_size, roi_size),
                sw_batch_size=sw_batch_size,
                predictor=model,
            )
            outputs = [post_pred(pred) for pred in decollate_batch(outputs)]
            metric(y_pred=outputs, y=labels)
    score = metric.aggregate().item()
    metric.reset()
    return score


def main() -> None:
    args = parse_args()
    is_distributed, rank, local_rank, world_size = setup_distributed(args)
    set_seed(args.seed, rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_ds, val_ds, data_keys, dataset_size = build_dataset_objects(args=args, rank=rank, is_distributed=is_distributed)
    train_loader, val_loader, train_sampler = build_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
    )

    model = build_model(device, is_distributed, local_rank)
    loss_fn = DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_dice = -1.0
    start_time = time.time()

    mlflow_client, run_context = setup_mlflow(args, rank)
    with run_context:
        if mlflow_client is not None:
            mlflow_client.log_params(
                {
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "num_samples": dataset_size,
                    "roi_size": args.roi_size,
                    "sw_batch_size": args.sw_batch_size,
                    "num_workers": args.num_workers,
                    "seed": args.seed,
                    "distributed": is_distributed,
                    "world_size": world_size,
                    "device": str(device),
                }
            )

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs = batch[data_keys[0]].to(device)
                labels = batch[data_keys[1]].to(device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            average_loss = running_loss / max(1, len(train_loader))
            val_dice = validate(
                model=model,
                val_loader=val_loader,
                device=device,
                roi_size=args.roi_size,
                sw_batch_size=args.sw_batch_size,
                data_keys=data_keys,
            )
            history.append({"epoch": epoch, "train_loss": average_loss, "val_dice": val_dice})

            if is_main_process(rank):
                print(
                    json.dumps(
                        {"epoch": epoch, "train_loss": round(average_loss, 4), "val_dice": round(val_dice, 4)},
                        sort_keys=True,
                    )
                )
                if mlflow_client is not None:
                    mlflow_client.log_metric("train_loss", average_loss, step=epoch)
                    mlflow_client.log_metric("val_dice", val_dice, step=epoch)

            if val_dice > best_dice:
                best_dice = val_dice
                if is_main_process(rank):
                    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save(state_dict, args.output_dir / "best_model.pt")

        elapsed = time.time() - start_time
        if is_main_process(rank):
            metrics = {
                "dataset": args.dataset,
                "world_size": world_size,
                "device": str(device),
                "epochs": args.epochs,
                "num_samples": dataset_size,
                "best_val_dice": best_dice,
                "elapsed_seconds": elapsed,
                "history": history,
            }
            metrics_path = args.output_dir / "metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=2)
            if mlflow_client is not None:
                mlflow_client.log_metric("best_val_dice", best_dice)
                mlflow_client.log_metric("elapsed_seconds", elapsed)
                mlflow_client.log_artifact(str(metrics_path))
                best_model_path = args.output_dir / "best_model.pt"
                if best_model_path.exists():
                    mlflow_client.log_artifact(str(best_model_path))
            print(f"Saved metrics to {metrics_path}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
