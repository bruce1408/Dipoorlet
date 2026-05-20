---
name: deploy
description: "Skill for the Deploy area of Dipoorlet. 24 symbols across 5 files."
---

# Deploy

24 symbols | 5 files | Cohesion: 86%

## When to Use

- Working with code in `dipoorlet/`
- Understanding how gen_stpu_minmax, quant_weight, quant_activation work
- Modifying deploy-related functionality

## Key Files

| File | Symbols |
|------|---------|
| `dipoorlet/deploy/deploy_stpu.py` | gen_stpu_minmax, quant_weight, quant_activation, merge_relu_layer, conv_wg_filter (+11) |
| `dipoorlet/deploy/deploy_rv.py` | step_zeropoint, gen_rv_yaml, gen1126, gen3568 |
| `dipoorlet/deploy/deploy_atlas.py` | get_step_zeropoint, gen_atlas_quant_param |
| `dipoorlet/deploy/deploy_base.py` | to_deploy |
| `dipoorlet/deploy/deploy_default.py` | deploy_dispatcher |

## Entry Points

Start here when exploring this area:

- **`gen_stpu_minmax`** (Function) — `dipoorlet/deploy/deploy_stpu.py:23`
- **`quant_weight`** (Function) — `dipoorlet/deploy/deploy_stpu.py:38`
- **`quant_activation`** (Function) — `dipoorlet/deploy/deploy_stpu.py:49`
- **`merge_relu_layer`** (Function) — `dipoorlet/deploy/deploy_stpu.py:65`
- **`conv_wg_filter`** (Function) — `dipoorlet/deploy/deploy_stpu.py:71`

## Key Symbols

| Symbol | Type | File | Line |
|--------|------|------|------|
| `gen_stpu_minmax` | Function | `dipoorlet/deploy/deploy_stpu.py` | 23 |
| `quant_weight` | Function | `dipoorlet/deploy/deploy_stpu.py` | 38 |
| `quant_activation` | Function | `dipoorlet/deploy/deploy_stpu.py` | 49 |
| `merge_relu_layer` | Function | `dipoorlet/deploy/deploy_stpu.py` | 65 |
| `conv_wg_filter` | Function | `dipoorlet/deploy/deploy_stpu.py` | 71 |
| `wg_weight_convt` | Function | `dipoorlet/deploy/deploy_stpu.py` | 83 |
| `conv_wg_layer` | Function | `dipoorlet/deploy/deploy_stpu.py` | 92 |
| `quant_bias` | Function | `dipoorlet/deploy/deploy_stpu.py` | 211 |
| `step_zeropoint` | Function | `dipoorlet/deploy/deploy_rv.py` | 10 |
| `gen_rv_yaml` | Function | `dipoorlet/deploy/deploy_rv.py` | 23 |
| `gen1126` | Function | `dipoorlet/deploy/deploy_rv.py` | 24 |
| `gen3568` | Function | `dipoorlet/deploy/deploy_rv.py` | 111 |
| `find_e` | Function | `dipoorlet/deploy/deploy_stpu.py` | 104 |
| `find_pool_ave_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 124 |
| `find_softmax_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 146 |
| `find_psroipooling_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 152 |
| `find_interp_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 119 |
| `find_conv_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 132 |
| `find_corr_emin` | Function | `dipoorlet/deploy/deploy_stpu.py` | 139 |
| `layer_emin_state` | Function | `dipoorlet/deploy/deploy_stpu.py` | 158 |

## Execution Flows

| Flow | Type | Steps |
|------|------|-------|
| `Gen_stpu_minmax → Conv_wg_filter` | intra_community | 3 |
| `Gen_stpu_minmax → Wg_weight_convt` | intra_community | 3 |
| `Gen_rv_yaml → Step_zeropoint` | intra_community | 3 |
| `Layer_emin_state → Find_e` | cross_community | 3 |

## How to Explore

1. `gitnexus_context({name: "gen_stpu_minmax"})` — see callers and callees
2. `gitnexus_query({query: "deploy"})` — find related execution flows
3. Read key files listed above for implementation details
