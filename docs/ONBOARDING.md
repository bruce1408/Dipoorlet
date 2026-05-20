# Dipoorlet Onboarding Guide

Welcome to Dipoorlet! This guide will help you understand the project structure, key components, and how to navigate the codebase.

## Project Overview

**Dipoorlet** is an offline quantization tool for ONNX models.

### What it does
- Performs activation calibration (Mse, Minmax, Hist)
- Applies weight transformation (BiasCorrection, WeightEqualization, Adaround, Brecq, QDrop)
- Generates quantized models for multiple hardware platforms

### Languages & Frameworks
- **Languages**: Python, Shell, Markdown, YAML
- **Frameworks**: PyTorch, ONNX, ONNXRuntime-GPU, ONNXSimplifier

### Supported Platforms
TensorRT, SNPE, STPU, Atlas, TI, RV, MagicMind, IMX

---

## Architecture Layers

Dipoorlet is organized into 11 architectural layers, flowing from CLI entry point through to platform-specific deployment:

| # | Layer | Description |
|---|-------|-------------|
| 1 | **CLI / Entry Point** | `dipoorlet/__main__.py` — Command-line interface, orchestrates the full pipeline |
| 2 | **Graph Utilities** | `dipoorlet/utils.py` — ONNXGraph class for model manipulation, shape inference, serialization |
| 3 | **Distributed Training** | `dipoorlet/dist_helper.py` — MPI/SLURM multi-GPU coordination |
| 4 | **Platform Configuration** | `dipoorlet/platform_settings.py` — Registry mapping deploy targets to quantization parameters |
| 5 | **Tensor Calibration** | `dipoorlet/tensor_cali/` — Activation range collection via ONNXRuntime (MinMax, Histogram, MSE) |
| 6 | **Forward Network** | `dipoorlet/forward_net.py` — ONNXRuntime inference engine with activation caching |
| 7 | **Quantization** | `dipoorlet/quantize.py` — Graph rewriting, injects Q/DQ nodes |
| 8 | **Weight Transformation** | `dipoorlet/weight_transform/` — Post-calibration weight optimizations |
| 9 | **Profiling** | `dipoorlet/profiling.py` — Quantization error analysis (cosine similarity, max-abs-gap) |
| 10 | **Deployment** | `dipoorlet/deploy/` — Platform-specific config writers (TRT/SNPE/QNN/RV/MagicMind/TI/IMX/Atlas/STPU) |
| 11 | **Shared Utilities** | `dipoorlet_utils/` — Dataset loading, batch generation |

### Pipeline Flow

```
__main__ → tensor_calibration → quantize.py (Q/DQ injection) → weight_calibration (bc/we/adaround/brecq) → profiling → deployment → platform-specific writer
```

---

## Guided Tour

Follow this 11-step walkthrough to learn the codebase:

### Step 1: Project Overview
Dipoorlet is an offline quantization tool for ONNX models. It supports activation calibration (Mse, Minmax, Hist), weight transformation (BiasCorrection, WeightEqualization), and SOTA finetune algorithms (Adaround, Brecq, QDrop). Generates quantitative parameters for multiple platforms (TensorRT, SNPE, STPU, ATLAS, etc.).

### Step 2: CLI Entry Point
The main CLI entry point invoked via `python -m dipoorlet`. Parses command-line arguments, initializes distributed training (MPI/SLURM), loads the ONNX model via ONNXGraph, drives the full quantization pipeline (calibration → weight transformation → profiling → deploy), and routes output to the target platform.
- **Key file**: `dipoorlet/__main__.py`

### Step 3: ONNX Graph Utilities
The ONNXGraph class is the core abstraction for model manipulation. It provides tensor shape/type management, initializer handling, node insertion/deletion, topological ordering, shape inference, quantization helper methods, and model serialization. Also provides calibration value management (save/load).
- **Key files**: `dipoorlet/utils.py` (contains `class:ONNXGraph`)

### Step 4: Distributed Training Coordination
Multi-GPU and multi-node coordination wrappers for MPI and SLURM environments. Handles distributed initialization, inter-process communication for calibration data reduction, and profiling result aggregation across ranks.
- **Key file**: `dipoorlet/dist_helper.py`

### Step 5: Platform Configuration
Central registry mapping deploy target (trt, snpe, qnn, rv, magicmind, ti, imx, atlas, stpu) to quantization parameters: bit-widths, symmetric/asymmetric mode, per-channel/per-tensor settings, which layers to quantize, and deploy-specific flags. Shapes the quantization strategy per target accelerator.
- **Key file**: `dipoorlet/platform_settings.py`

### Step 6: Tensor Calibration (Activation Quantization)
Collects activation ranges from calibration data via ONNXRuntime forward passes. Implements three calibration algorithms: MinMax (forward_get_minmax), Histogram with KL thresholding (forward_get_hist), and MSE/OCTAV (forward_net_octav). Includes weight-only minmax clipping. Supports both classic CNN and transformer architectures.
- **Key files**: `dipoorlet/tensor_cali/tensor_cali_base.py`, `dipoorlet/tensor_cali/basic_algorithm.py`, `dipoorlet/forward_net.py`

### Step 7: Quantization (Graph Rewriting)
Injects Q/DQ (QuantizeLinear/DequantizeLinear) fake-quantization nodes into the ONNX graph, setting up scale/zero-point tensors. Handles per-channel and per-layer quantization, symmetric and asymmetric modes, ConvTranspose channel-last transpose, ReLU merging, and TRT-specific Add handling.
- **Key file**: `dipoorlet/quantize.py`

### Step 8: Weight Transformation
Post-calibration weight transformations that improve quantized model accuracy. Includes BiasCorrection (corrects bias after quantization), WeightEqualization (equalizes per-channel ranges), Adaround (activation-based range optimization for weight rounding), Brecq/QDrop (bit-level reconstruction), sparse quantization, and BN update. Orchestrated by weight_trans_base.py.
- **Key files**: `dipoorlet/weight_transform/weight_trans_base.py`, `dipoorlet/weight_transform/adaround.py`, `dipoorlet/weight_transform/brecq.py`, `dipoorlet/weight_transform/bias_correction.py`, `dipoorlet/weight_transform/weight_equalization.py`, `dipoorlet/weight_transform/sparse_quant.py`, `dipoorlet/weight_transform/sparse_quant_layer.py`, `dipoorlet/weight_transform/ada_quant_layer.py`, `dipoorlet/weight_transform/update_bn.py`, `dipoorlet/weight_transform/utils.py`

### Step 9: Profiling (Quantization Error Analysis)
Layer-wise and model-wise quantization error profiling using cosine similarity and max-abs-gap metrics. Supports multipass CNN profiling (quantize_profiling_multipass), transformer profiling with activation caching (quantize_profiling_transformer), and layerwise sensitivity analysis (quantize_profiling_layerwise). Outputs per-layer and aggregated profiling JSON.
- **Key file**: `dipoorlet/profiling.py`

### Step 10: Deployment (Platform-Specific Export)
Writes quantized model and range files in each target accelerator's format. Base dispatcher (deploy_base.py) routes to platform-specific generators: TensorRT, SNPE, QNN, RV, MagicMind, TI, IMX, Atlas, STPU. Each deploy_*.py registers with the dispatcher and writes config blobs and model files tailored to the target.
- **Key files**: `dipoorlet/deploy/deploy_base.py`, `dipoorlet/deploy/deploy_default.py`, and platform-specific files

### Step 11: Shared Utilities
Dataset loading (binary calibration files) and calibration batch generation used across calibration, weight transformation, and profiling. The calibrator.py module provides activation range collectors; dataset.py handles binary data loading and batch formation.
- **Key files**: `dipoorlet_utils/calibrator.py`, `dipoorlet_utils/dataset.py`

---

## File Map

### Core dipoorlet package

| File | Purpose |
|------|---------|
| `dipoorlet/__main__.py` | CLI entry point, orchestrates full pipeline |
| `dipoorlet/utils.py` | ONNXGraph class — model manipulation, shape inference, serialization |
| `dipoorlet/dist_helper.py` | MPI/SLURM distributed coordination |
| `dipoorlet/platform_settings.py` | Deploy target → quantization params registry |
| `dipoorlet/forward_net.py` | ONNXRuntime inference engine, activation caching |
| `dipoorlet/quantize.py` | Q/DQ node injection, graph rewriting |
| `dipoorlet/profiling.py` | Quantization error profiling (cosine similarity, max-abs-gap) |

### tensor_cali/ — Activation Calibration

| File | Purpose |
|------|---------|
| `dipoorlet/tensor_cali/__init__.py` | Barrel module, exports `tensor_calibration` and `find_clip_value` |
| `dipoorlet/tensor_cali/tensor_cali_base.py` | Core calibration dispatcher |
| `dipoorlet/tensor_cali/basic_algorithm.py` | MinMax, Histogram (KL), MSE/OCTAV implementations |

### weight_transform/ — Weight Optimization

| File | Purpose |
|------|---------|
| `dipoorlet/weight_transform/__init__.py` | Barrel module |
| `dipoorlet/weight_transform/weight_trans_base.py` | Orchestrates weight calibration pipeline |
| `dipoorlet/weight_transform/adaround.py` | ADAROUND algorithm |
| `dipoorlet/weight_transform/brecq.py` | BRECQ algorithm with Q-drop support |
| `dipoorlet/weight_transform/bias_correction.py` | Bias correction |
| `dipoorlet/weight_transform/weight_equalization.py` | Weight equalization |
| `dipoorlet/weight_transform/sparse_quant.py` | Sparse quantization |
| `dipoorlet/weight_transform/ada_quant_layer.py` | AdaRound layer implementation |
| `dipoorlet/weight_transform/update_bn.py` | BatchNorm update |

### deploy/ — Platform Export

| File | Purpose |
|------|---------|
| `dipoorlet/deploy/deploy_base.py` | Base dispatcher |
| `dipoorlet/deploy/deploy_default.py` | Default implementation |
| `dipoorlet/deploy/deploy_trt.py` | TensorRT |
| `dipoorlet/deploy/deploy_snpe.py` | Snapdragon Neural Processing Engine |
| `dipoorlet/deploy/deploy_qnn.py` | QNN (Qualcomm) |
| `dipoorlet/deploy/deploy_rv.py` | RV (RISC-V?) |
| `dipoorlet/deploy/deploy_magicmind.py` | MagicMind |
| `dipoorlet/deploy/deploy_ti.py` | Texas Instruments |
| `dipoorlet/deploy/deploy_imx.py` | NXP i.MX |
| `dipoorlet/deploy/deploy_atlas.py` | Atlas |
| `dipoorlet/deploy/deploy_stpu.py` | STPU |

### dipoorlet_utils/ — Shared

| File | Purpose |
|------|---------|
| `dipoorlet_utils/calibrator.py` | TensorRT calibration utilities, CalibDataLoader |
| `dipoorlet_utils/dataset.py` | Image dataset loading (TinyImageNet support) |

---

## Complexity Hotspots

The following files have `moderate` complexity and should be approached carefully:

| File | Notes |
|------|-------|
| `dipoorlet/__main__.py` | CLI orchestration, many branches |
| `dipoorlet/utils.py` | ONNXGraph core, extensive graph operations |
| `dipoorlet/quantize.py` | Q/DQ injection, many special cases (ConvTranspose, ReLU, TRT Add) |
| `dipoorlet/profiling.py` | Multiple profiling modes (multipass, transformer, layerwise) |
| `dipoorlet/forward_net.py` | ONNXRuntime wrapper, activation caching logic |
| `dipoorlet/tensor_cali/tensor_cali_base.py` | Calibration dispatcher, multiple algorithm paths |
| `dipoorlet/tensor_cali/basic_algorithm.py` | MinMax, Histogram, MSE/OCTAV — complex dispatch table |
| `dipoorlet/weight_transform/weight_trans_base.py` | Pipeline orchestration, multiple weight algorithms |
| `dipoorlet/weight_transform/adaround.py` | ADAROUND optimization loop |
| `dipoorlet/weight_transform/brecq.py` | BRECQ with Q-drop support |
| `dipoorlet/deploy/deploy_base.py` | Dispatcher routing to 9 platform targets |
| `dipoorlet/deploy/deploy_trt.py` | TensorRT-specific writing logic |

---

## Quick Start

```bash
# Run quantization on MobileNetV2
python -m dipoorlet --model resnet18.onnx --dataset ./calib_data/ --output ./output/

# With weight transformation (Adaround)
python -m dipoorlet --model resnet18.onnx --dataset ./calib_data/ --weight_trans adaround --output ./output/

# Deploy to TensorRT
python -m dipoorlet --model resnet18.onnx --dataset ./calib_data/ --deploy trt --output ./output/
```

See `dipoorlet/__main__.py` for full argument reference.

---

## Key Concepts

- **Q/DQ Nodes**: Fake quantization nodes (QuantizeLinear/DequantizeLinear) inserted into ONNX graph
- **Activation Calibration**: Collecting activation ranges from calibration data (MinMax, Histogram, MSE)
- **Weight Transformation**: Post-calibration optimizations (BiasCorrection, WeightEqualization, Adaround, Brecq)
- **Per-channel vs Per-tensor**: Quantization granularity — per-channel preserves accuracy better for weights
- **Symmetric vs Asymmetric**: Scale/zero-point computation mode
- **ONNXGraph**: Core abstraction for all ONNX model manipulations in this project
