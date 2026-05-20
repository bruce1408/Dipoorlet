---
name: dipoorlet-utils
description: "Skill for the Dipoorlet_utils area of Dipoorlet. 16 symbols across 5 files."
---

# Dipoorlet_utils

16 symbols | 5 files | Cohesion: 87%

## When to Use

- Working with code in `dipoorlet_utils/`
- Understanding how evaluate_model, evaluate_model, evaluate_model work
- Modifying dipoorlet_utils-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `dipoorlet_utils/calibrator.py` | get_calib_data_path, Preprocess, __init__, next_batch, generate_dipoorlet_calib (+3) |
| `dipoorlet_utils/dataset.py` | evaluate, get_dataloaders, __init__, _find_classes, _make_dataset |
| `DemoLab/2_mobile_v2_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py` | evaluate_model |
| `DemoLab/3_quant_resnet18/3_1_resnet18_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py` | evaluate_model |
| `DemoLab/3_quant_resnet18/3_4_resnet18_dipoorlet_qnn/1_1_evaluate_model_and_export_onnx.py` | evaluate_model |

## Entry Points

Start here when exploring this area:

- **`evaluate_model`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py:20`
- **`evaluate_model`** (Function) — `DemoLab/3_quant_resnet18/3_1_resnet18_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py:42`
- **`evaluate_model`** (Function) — `DemoLab/3_quant_resnet18/3_4_resnet18_dipoorlet_qnn/1_1_evaluate_model_and_export_onnx.py:43`
- **`get_dataloaders`** (Function) — `dipoorlet_utils/dataset.py:245`
- **`get_calib_data_path`** (Function) — `dipoorlet_utils/calibrator.py:38`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `evaluate_model` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py` | 20 |
| `evaluate_model` | Function | `DemoLab/3_quant_resnet18/3_1_resnet18_dipoorlet_trt/2_1_evaluate_model_and_export_onnx.py` | 42 |
| `evaluate_model` | Function | `DemoLab/3_quant_resnet18/3_4_resnet18_dipoorlet_qnn/1_1_evaluate_model_and_export_onnx.py` | 43 |
| `get_dataloaders` | Function | `dipoorlet_utils/dataset.py` | 245 |
| `get_calib_data_path` | Function | `dipoorlet_utils/calibrator.py` | 38 |
| `Preprocess` | Function | `dipoorlet_utils/calibrator.py` | 52 |
| `generate_dipoorlet_calib` | Function | `dipoorlet_utils/calibrator.py` | 126 |
| `LetterBox` | Function | `dipoorlet_utils/calibrator.py` | 140 |
| `yolov8_process` | Function | `dipoorlet_utils/calibrator.py` | 161 |
| `get_yolov8_calib` | Function | `dipoorlet_utils/calibrator.py` | 173 |
| `evaluate` | Method | `dipoorlet_utils/dataset.py` | 94 |
| `next_batch` | Method | `dipoorlet_utils/calibrator.py` | 78 |
| `__init__` | Method | `dipoorlet_utils/calibrator.py` | 67 |
| `__init__` | Method | `dipoorlet_utils/dataset.py` | 193 |
| `_find_classes` | Method | `dipoorlet_utils/dataset.py` | 214 |
| `_make_dataset` | Method | `dipoorlet_utils/dataset.py` | 221 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Get_yolov8_calib → LetterBox` | intra_community | 3 |
| `Main → Get_dataloaders` | cross_community | 3 |

## How to Explore

1. `gitnexus_context({name: "evaluate_model"})` — see callers and callees
2. `gitnexus_query({query: "dipoorlet_utils"})` — find related execution flows
3. Read key files listed above for implementation details
