---
name: 3-5-resnet18-aimet-qnn
description: "Skill for the 3_5_resnet18_aimet_qnn area of Dipoorlet. 7 symbols across 1 files."
---

# 3_5_resnet18_aimet_qnn

7 symbols | 1 files | Cohesion: 92%

## When to Use

- Working with code in `DemoLab/`
- Understanding how allocate_buffers, do_inference, postprocess_the_outputs work
- Modifying 3_5_resnet18_aimet_qnn-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | allocate_buffers, do_inference, postprocess_the_outputs, deserializing_engine, calculate_tensorrt_acc (+2) |

## Entry Points

Start here when exploring this area:

- **`allocate_buffers`** (Function) — `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py:37`
- **`do_inference`** (Function) — `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py:62`
- **`postprocess_the_outputs`** (Function) — `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py:80`
- **`deserializing_engine`** (Function) — `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py:85`
- **`calculate_tensorrt_acc`** (Function) — `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py:92`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `allocate_buffers` | Function | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 37 |
| `do_inference` | Function | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 62 |
| `postprocess_the_outputs` | Function | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 80 |
| `deserializing_engine` | Function | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 85 |
| `calculate_tensorrt_acc` | Function | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 92 |
| `__str__` | Method | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 30 |
| `__repr__` | Method | `DemoLab/3_quant_resnet18/3_5_resnet18_aimet_qnn/calculate_trt_engine_acc.py` | 33 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Dipoorlet_utils | 1 calls |

## How to Explore

1. `gitnexus_context({name: "allocate_buffers"})` — see callers and callees
2. `gitnexus_query({query: "3_5_resnet18_aimet_qnn"})` — find related execution flows
3. Read key files listed above for implementation details
