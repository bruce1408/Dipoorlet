---
name: 2-mobile-v2-dipoorlet-trt
description: "Skill for the 2_mobile_v2_dipoorlet_trt area of Dipoorlet. 25 symbols across 5 files."
---

# 2_mobile_v2_dipoorlet_trt

25 symbols | 5 files | Cohesion: 95%

## When to Use

- Working with code in `DemoLab/`
- Understanding how get_config, find_latest_checkpoint, mixup_data work
- Modifying 2_mobile_v2_dipoorlet_trt-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | get_config, find_latest_checkpoint, mixup_data, mixup_criterion, update (+4) |
| `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | allocate_buffers, do_inference, postprocess_the_outputs, deserializing_engine, main (+2) |
| `DemoLab/2_mobile_v2_dipoorlet_trt/1_2_train_customed_model_basic.py` | get_config, find_latest_checkpoint, train_model, main |
| `DemoLab/2_mobile_v2_dipoorlet_trt/5_build_trt_engine_use_dipoorlet.py` | set_dynamic_range, buildEngine, main |
| `DemoLab/2_mobile_v2_dipoorlet_trt/3_1_trt_quant_build_trt_engine_deplpy_trt.py` | buildEngine, main |

## Entry Points

Start here when exploring this area:

- **`get_config`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py:21`
- **`find_latest_checkpoint`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py:43`
- **`mixup_data`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py:71`
- **`mixup_criterion`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py:86`
- **`train_model`** (Function) — `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py:123`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `get_config` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 21 |
| `find_latest_checkpoint` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 43 |
| `mixup_data` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 71 |
| `mixup_criterion` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 86 |
| `train_model` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 123 |
| `main` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_3_train_customed_model_advance.py` | 315 |
| `allocate_buffers` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | 37 |
| `do_inference` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | 63 |
| `postprocess_the_outputs` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | 78 |
| `deserializing_engine` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | 83 |
| `main` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/6_test_quant_acc.py` | 90 |
| `get_config` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_2_train_customed_model_basic.py` | 18 |
| `find_latest_checkpoint` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_2_train_customed_model_basic.py` | 41 |
| `train_model` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_2_train_customed_model_basic.py` | 53 |
| `main` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/1_2_train_customed_model_basic.py` | 204 |
| `set_dynamic_range` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/5_build_trt_engine_use_dipoorlet.py` | 18 |
| `buildEngine` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/5_build_trt_engine_use_dipoorlet.py` | 46 |
| `main` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/5_build_trt_engine_use_dipoorlet.py` | 68 |
| `buildEngine` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/3_1_trt_quant_build_trt_engine_deplpy_trt.py` | 27 |
| `main` | Function | `DemoLab/2_mobile_v2_dipoorlet_trt/3_1_trt_quant_build_trt_engine_deplpy_trt.py` | 63 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Main → Apply_shadow` | intra_community | 3 |
| `Main → Update` | intra_community | 3 |
| `Main → Restore` | intra_community | 3 |
| `Main → Mixup_data` | intra_community | 3 |
| `Main → Set_dynamic_range` | intra_community | 3 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Dipoorlet_utils | 2 calls |

## How to Explore

1. `gitnexus_context({name: "get_config"})` — see callers and callees
2. `gitnexus_query({query: "2_mobile_v2_dipoorlet_trt"})` — find related execution flows
3. Read key files listed above for implementation details
