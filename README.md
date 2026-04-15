# MONAI Portfolio Demo

Reproducible 3D medical image segmentation workflow using `MONAI`, `PyTorch`, and `MLflow`, with an HPC-oriented distributed training path and `SLURM` launch template.

This project is a small, reproducible medical image segmentation demo built on top of [MONAI](https://github.com/Project-MONAI/MONAI). It is designed to showcase the parts of the job description that are realistic to demonstrate in a portfolio:

- PyTorch and MONAI-based 3D medical image segmentation
- reproducible training and evaluation workflows
- sliding-window inference and Dice-based validation
- optional MLflow experiment tracking
- optional multi-GPU execution with PyTorch Distributed Data Parallel
- an HPC-oriented SLURM launch path

The training script uses synthetic 3D NIfTI volumes so the workflow is runnable without downloading a large clinical dataset first.

## Overview

This project demonstrates how to build and track a compact 3D medical image segmentation workflow using `MONAI`, `PyTorch`, and `MLflow`. The implementation focuses on reproducibility, experiment tracking, and HPC-oriented execution patterns rather than on claiming a novel model contribution.

Core capabilities demonstrated here:

- 3D volumetric segmentation with MONAI transforms and a UNet-based model
- synthetic NIfTI data generation for fast, reproducible local testing
- support for a real public dataset via Medical Segmentation Decathlon Spleen
- Dice-based validation and sliding-window inference
- MLflow logging for parameters, metrics, runtime, and artifacts
- optional distributed execution with `torchrun`
- a `SLURM` submission template for shared GPU environments

## Why This Repo Matters

This repo is designed to be easy for recruiters and hiring managers to scan. It shows practical experience with:

- medical image analysis workflows
- PyTorch-based model development
- experiment tracking and reproducibility
- Linux and HPC-oriented execution patterns
- code organized as a small, runnable research engineering project

## What Is Mine

This repo folder contains my portfolio scaffold:

- [`train_segmentation_demo.py`](./train_segmentation_demo.py): a compact segmentation training and evaluation workflow
- [`slurm/train_monai_ddp.slurm`](./slurm/train_monai_ddp.slurm): example batch job for cluster execution
- this README and run notes

The implementation is informed by official MONAI examples, especially:

- `tutorials/acceleration/distributed_training/unet_training_ddp.py`
- `tutorials/3d_segmentation/torch/unet_training_dict.py`

## Local Run

Create an environment with PyTorch and MONAI installed, then run:

```bash
python -m pip install -r requirements.txt
python train_segmentation_demo.py --epochs 2 --output-dir artifacts/run/local
```

This default command uses the synthetic dataset path for a fast local sanity check.

Expected outputs:

- `artifacts/run/local/best_model.pt`
- `artifacts/run/local/metrics.json`

To also track the run in MLflow:

```bash
python train_segmentation_demo.py \
  --epochs 2 \
  --output-dir artifacts/run/local \
  --mlflow \
  --mlflow-experiment monai-portfolio-demo
```

If you want the local file-based MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Real Dataset Run

The script also supports the public Medical Segmentation Decathlon Spleen dataset through MONAI's `DecathlonDataset`.

Example:

```bash
python train_segmentation_demo.py \
  --dataset msd_spleen \
  --data-dir artifacts/msd_spleen \
  --epochs 2 \
  --output-dir artifacts/run/msd_spleen \
  --mlflow \
  --mlflow-experiment monai-portfolio-demo
```

Notes:

- the first run downloads and prepares the dataset automatically
- this path is more representative of real medical imaging experimentation than the synthetic demo
- on CPU it may be slow; a GPU-enabled environment is strongly preferred

## Results

This project logs segmentation experiments in MLflow using a local SQLite-backed tracking store. In a 2-epoch CPU run on synthetic 3D NIfTI volumes:

- training loss decreased from `0.6408` to `0.5877`
- validation Dice improved from `0.2637` to `0.5235`
- total runtime was approximately `57.5` seconds

The workflow also saves reproducible artifacts, including:

- `metrics.json`
- `best_model.pt`

In a 2-epoch CPU run on the real Medical Segmentation Decathlon Spleen dataset:

- training loss decreased from `0.8993` to `0.8918`
- validation Dice improved from `0.0265` to `0.0367`

These real-dataset numbers are intentionally presented as an early proof-of-work run, not as a tuned benchmark. The main value is demonstrating a functioning MONAI pipeline on a public medical imaging dataset with experiment tracking and reproducible outputs.

Recommended README screenshot:

- MLflow run details page showing parameters, metrics, and model artifacts

Suggested GitHub screenshot placement:

- add the MLflow run-details screenshot directly below this section

## Multi-GPU Run

On a workstation with multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train_segmentation_demo.py --distributed --epochs 2
```

## HPC / SLURM Run

The included SLURM script is a template for single-node multi-GPU execution:

```bash
sbatch slurm/train_monai_ddp.slurm
```

Things I would adapt on a real cluster:

- account, partition, and module lines
- container or conda environment activation
- dataset staging to scratch storage
- logging GPU, memory, and CPU utilization with tools such as `nvidia-smi`, `sstat`, and cluster profiling utilities
- pointing MLflow at a shared tracking server for team-visible experiment history
- adjusting storage paths to use shared scratch or project storage for larger imaging datasets

