---
name: 3-3-resnet18-native-int8-qnn
description: "Skill for the 3_3_resnet18_native_int8_qnn area of Dipoorlet. 10 symbols across 2 files."
---

# 3_3_resnet18_native_int8_qnn

10 symbols | 2 files | Cohesion: 100%

## When to Use

- Working with code in `DemoLab/`
- Understanding how load_onnx_model, analyze_conv_layers, analyze_weight_distribution work
- Modifying 3_3_resnet18_native_int8_qnn-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | load_onnx_model, analyze_conv_layers, analyze_weight_distribution, analyze_bias_distribution, save_results_to_json (+3) |
| `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/5_1_parse_raw_res.py` | preprocess_image, infer_with_onnx |

## Entry Points

Start here when exploring this area:

- **`load_onnx_model`** (Function) — `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py:15`
- **`analyze_conv_layers`** (Function) — `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py:32`
- **`analyze_weight_distribution`** (Function) — `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py:80`
- **`analyze_bias_distribution`** (Function) — `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py:143`
- **`save_results_to_json`** (Function) — `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py:187`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `load_onnx_model` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 15 |
| `analyze_conv_layers` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 32 |
| `analyze_weight_distribution` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 80 |
| `analyze_bias_distribution` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 143 |
| `save_results_to_json` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 187 |
| `convert_numpy` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 190 |
| `generate_summary_report` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 209 |
| `main` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/read_conv_param.py` | 243 |
| `preprocess_image` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/5_1_parse_raw_res.py` | 52 |
| `infer_with_onnx` | Function | `DemoLab/3_quant_resnet18/3_3_resnet18_native_int8_qnn/5_1_parse_raw_res.py` | 96 |

## How to Explore

1. `gitnexus_context({name: "load_onnx_model"})` — see callers and callees
2. `gitnexus_query({query: "3_3_resnet18_native_int8_qnn"})` — find related execution flows
3. Read key files listed above for implementation details
