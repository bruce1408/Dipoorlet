---
name: weight-transform
description: "Skill for the Weight_transform area of Dipoorlet. 42 symbols across 10 files."
---

# Weight_transform

42 symbols | 10 files | Cohesion: 86%

## When to Use

- Working with code in `dipoorlet/`
- Understanding how quant_graph, update_conv_node_bias, bias_correction work
- Modifying weight_transform-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `dipoorlet/weight_transform/ada_quant_layer.py` | nnie_rest_init, zero_point, quant_acti, quant_weight, quant_weight_nnie (+9) |
| `dipoorlet/weight_transform/sparse_quant_layer.py` | quant_weight_wo_roundmask, create_unstruction_mask, create_nv24_mask, prune_weight, forward (+5) |
| `dipoorlet/weight_transform/weight_equalization.py` | weight_equalization, converged, find_successor, node_has_equalized |
| `dipoorlet/utils.py` | save_onnx_model, update_model, copy_from |
| `dipoorlet/weight_transform/update_bn.py` | update_bn_node, update_bn_multipass, update_bn_onepass |
| `dipoorlet/weight_transform/bias_correction.py` | update_conv_node_bias, bias_correction |
| `dipoorlet/weight_transform/adaround.py` | adaround, learning_round_mask |
| `dipoorlet/weight_transform/brecq.py` | brecq, learning_round_mask |
| `dipoorlet/quantize.py` | quant_graph |
| `dipoorlet/weight_transform/utils.py` | update_weight |

## Entry Points

Start here when exploring this area:

- **`quant_graph`** (Function) — `dipoorlet/quantize.py:19`
- **`update_conv_node_bias`** (Function) — `dipoorlet/weight_transform/bias_correction.py:8`
- **`bias_correction`** (Function) — `dipoorlet/weight_transform/bias_correction.py:33`
- **`update_bn_node`** (Function) — `dipoorlet/weight_transform/update_bn.py:11`
- **`update_bn_multipass`** (Function) — `dipoorlet/weight_transform/update_bn.py:25`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `quant_graph` | Function | `dipoorlet/quantize.py` | 19 |
| `update_conv_node_bias` | Function | `dipoorlet/weight_transform/bias_correction.py` | 8 |
| `bias_correction` | Function | `dipoorlet/weight_transform/bias_correction.py` | 33 |
| `update_bn_node` | Function | `dipoorlet/weight_transform/update_bn.py` | 11 |
| `update_bn_multipass` | Function | `dipoorlet/weight_transform/update_bn.py` | 25 |
| `update_bn_onepass` | Function | `dipoorlet/weight_transform/update_bn.py` | 50 |
| `update_weight` | Function | `dipoorlet/weight_transform/utils.py` | 23 |
| `weight_equalization` | Function | `dipoorlet/weight_transform/weight_equalization.py` | 42 |
| `converged` | Function | `dipoorlet/weight_transform/weight_equalization.py` | 101 |
| `nnie_rest_init` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 11 |
| `zero_point` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 13 |
| `quant_acti` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 30 |
| `quant_weight` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 41 |
| `quant_weight_nnie` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 55 |
| `quant_acti_nnie` | Function | `dipoorlet/weight_transform/ada_quant_layer.py` | 78 |
| `adaround` | Function | `dipoorlet/weight_transform/adaround.py` | 27 |
| `learning_round_mask` | Function | `dipoorlet/weight_transform/adaround.py` | 152 |
| `brecq` | Function | `dipoorlet/weight_transform/brecq.py` | 19 |
| `learning_round_mask` | Function | `dipoorlet/weight_transform/brecq.py` | 161 |
| `find_successor` | Function | `dipoorlet/weight_transform/weight_equalization.py` | 14 |

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
| Dipoorlet | 6 calls |

## How to Explore

1. `gitnexus_context({name: "quant_graph"})` — see callers and callees
2. `gitnexus_query({query: "weight_transform"})` — find related execution flows
3. Read key files listed above for implementation details
