# MONAI Portfolio Demo

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
- Dice-based validation and sliding-window inference
- MLflow logging for parameters, metrics, runtime, and artifacts
- optional distributed execution with `torchrun`
- a `SLURM` submission template for shared GPU environments

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

## Results

This project logs segmentation experiments in MLflow using a local SQLite-backed tracking store. In a 2-epoch CPU run on synthetic 3D NIfTI volumes:

- training loss decreased from `0.6408` to `0.5877`
- validation Dice improved from `0.2637` to `0.5235`
- total runtime was approximately `57.5` seconds

The workflow also saves reproducible artifacts, including:

- `metrics.json`
- `best_model.pt`

Recommended README screenshot:

- MLflow run details page showing parameters, metrics, and model artifacts

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

## Resume / Portfolio Framing

Here is an honest way to describe this project:

> Built a reproducible 3D medical image segmentation workflow with MONAI and PyTorch, including MLflow experiment tracking, Dice-based evaluation, artifact logging, and an HPC-oriented multi-GPU/SLURM execution path.

You can also expand it for an interview:

- generated synthetic NIfTI volumes to make the workflow easy to reproduce
- used MONAI transforms and UNet components to build a compact 3D segmentation pipeline
- added MLflow-based experiment tracking for parameters, metrics, and model artifacts
- added a PyTorch DDP path to mirror multi-GPU training patterns used on HPC systems
- prepared a SLURM submission template to show how the same workflow can be adapted to shared GPU clusters

## Suggested Next Steps

To move this closer to the target JD, I would add one or two of these next:

- swap the synthetic dataset for a public dataset such as MSD Spleen or BTCV
- benchmark runs in a shared MLflow tracking server or TensorBoard instance
- containerize the run with Apptainer or Docker
- benchmark throughput and utilization across 1 GPU vs. multi-GPU runs
- add uncertainty estimation or calibration metrics for model quality analysis

The highest-value next step is replacing the synthetic dataset with a real public medical imaging dataset. That would make the project much stronger for research-oriented AI and HPC roles.
