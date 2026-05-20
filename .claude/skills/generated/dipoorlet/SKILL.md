---
name: dipoorlet
description: "Skill for the Dipoorlet area of Dipoorlet. 55 symbols across 7 files."
---

# Dipoorlet

55 symbols | 7 files | Cohesion: 78%

## When to Use

- Working with code in `dipoorlet/`
- Understanding how forward_get_tensor, update_node_quant_profiling, quantize_profiling_multipass work
- Modifying dipoorlet-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `dipoorlet/utils.py` | __init__, set_names, convert_constant_to_init, prepare_initializer, get_inp_oup (+18) |
| `dipoorlet/forward_net.py` | __init__, reset, fetch_input, input_generator, __getitem__ (+10) |
| `dipoorlet/profiling.py` | update_node_quant_profiling, quantize_profiling_multipass, quantize_profiling_transformer, update_quant_model_cosine, get_output_single_map (+2) |
| `dipoorlet/quantize.py` | delete_fake_quant_node, get_input_idx, insert_fake_quant_node, insert_fake_quant_node_output, get_qnode_by_param (+1) |
| `dipoorlet/weight_transform/sparse_quant.py` | sparse_quant, learning_sparse_quant |
| `dipoorlet/weight_transform/weight_trans_base.py` | weight_calibration |
| `dipoorlet/weight_transform/utils.py` | get_quant_tensor |

## Entry Points

Start here when exploring this area:

- **`forward_get_tensor`** (Function) — `dipoorlet/forward_net.py:466`
- **`update_node_quant_profiling`** (Function) — `dipoorlet/profiling.py:23`
- **`quantize_profiling_multipass`** (Function) — `dipoorlet/profiling.py:41`
- **`quantize_profiling_transformer`** (Function) — `dipoorlet/profiling.py:109`
- **`update_quant_model_cosine`** (Function) — `dipoorlet/profiling.py:166`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `forward_get_tensor` | Function | `dipoorlet/forward_net.py` | 466 |
| `update_node_quant_profiling` | Function | `dipoorlet/profiling.py` | 23 |
| `quantize_profiling_multipass` | Function | `dipoorlet/profiling.py` | 41 |
| `quantize_profiling_transformer` | Function | `dipoorlet/profiling.py` | 109 |
| `update_quant_model_cosine` | Function | `dipoorlet/profiling.py` | 166 |
| `get_output_single_map` | Function | `dipoorlet/profiling.py` | 209 |
| `show_model_profiling_res` | Function | `dipoorlet/profiling.py` | 252 |
| `cos_similarity` | Function | `dipoorlet/utils.py` | 290 |
| `update_model_path` | Function | `dipoorlet/utils.py` | 327 |
| `save_clip_val` | Function | `dipoorlet/utils.py` | 334 |
| `reduce_clip_val` | Function | `dipoorlet/utils.py` | 347 |
| `load_clip_val` | Function | `dipoorlet/utils.py` | 369 |
| `sparse_quant` | Function | `dipoorlet/weight_transform/sparse_quant.py` | 18 |
| `learning_sparse_quant` | Function | `dipoorlet/weight_transform/sparse_quant.py` | 106 |
| `weight_calibration` | Function | `dipoorlet/weight_transform/weight_trans_base.py` | 14 |
| `forward_get_output` | Function | `dipoorlet/forward_net.py` | 488 |
| `quantize_profiling_layerwise` | Function | `dipoorlet/profiling.py` | 273 |
| `delete_fake_quant_node` | Function | `dipoorlet/quantize.py` | 111 |
| `get_input_idx` | Function | `dipoorlet/quantize.py` | 113 |
| `max_abs_gap` | Function | `dipoorlet/utils.py` | 297 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Update_bn_multipass → Make_quant_dequant` | cross_community | 5 |
| `Quantize_profiling_layerwise → Make_quant_dequant` | cross_community | 5 |
| `Bias_correction → Make_quant_dequant` | cross_community | 5 |
| `Quantize_profiling_multipass → Make_quant_dequant` | cross_community | 5 |
| `Adaround → Make_quant_dequant` | cross_community | 5 |
| `Brecq → Make_quant_dequant` | cross_community | 5 |
| `Sparse_quant → Make_quant_dequant` | cross_community | 5 |
| `Update_bn_onepass → Make_quant_dequant` | cross_community | 5 |
| `Quantize_profiling_transformer → Make_quant_dequant` | cross_community | 5 |
| `Weight_calibration → Set_index` | cross_community | 4 |

## Connected Areas

| Area | Connections |
|------|-------------|
| Weight_transform | 10 calls |

## How to Explore

1. `gitnexus_context({name: "forward_get_tensor"})` — see callers and callees
2. `gitnexus_query({query: "dipoorlet"})` — find related execution flows
3. Read key files listed above for implementation details
