# Dipoorlet Architecture

> Generated from GitNexus knowledge graph — 2,425 symbols, 74 execution flows, 11 functional areas.

## Overview

Dipoorlet is an offline quantization tool for ONNX models. It takes a pre-trained ONNX model and a calibration dataset, then produces platform-specific quantization parameters (scale, zero_point, min/max ranges) for deployment to 9 inference engines.

**Key design**: Operates exclusively on the ONNX graph representation — no framework-native model objects. ONNX Runtime (CUDA) serves as the forward engine.

---

## 1. System Architecture

```mermaid
graph TB
    subgraph Entry["入口层"]
        CLI["__main__.py<br/>argparse CLI · 分布式初始化 · Pipeline 编排"]
    end

    subgraph Core["核心引擎层"]
        GRAPH["ONNXGraph<br/>ONNX 图抽象 · 拓扑管理 · 初始化器读写"]
        FWD["forward_net.py<br/>ort.InferenceSession · ActivationCache<br/>全图推理 / 子图惰性推理"]
        PLAT["platform_settings.py<br/>9 后端策略配置表<br/>quant_nodes · qw/qi_params"]
    end

    subgraph Business["业务逻辑层"]
        TCAL["tensor_cali/<br/>激活值校准<br/>MSE · Hist · MinMax"]
        WT["weight_transform/<br/>权重变换管道<br/>BC · WE · BN · AdaRound · BrecQ · Sparse"]
        QGRAPH["quantize.py<br/>QDQ 图构建<br/>QuantizeLinear + DequantizeLinear"]
        PROF["profiling.py<br/>量化误差分析<br/>Cosine · MaxAbsGap · Layerwise"]
    end

    subgraph Export["导出层"]
        DEPLOY["deploy/<br/>平台量化参数生成<br/>TRT · SNPE · QNN · STPU<br/>Atlas · TI · i.MX · RV · MagicMind"]
    end

    subgraph Infra["基础设施"]
        DIST["dist_helper.py<br/>MPI · Slurm · NCCL"]
        UTILS["dipoorlet_utils/<br/>校准数据集构建"]
    end

    CLI --> GRAPH
    CLI --> FWD
    CLI --> PLAT
    CLI --> TCAL
    CLI --> WT
    CLI --> QGRAPH
    CLI --> PROF
    CLI --> DEPLOY
    CLI --> DIST

    TCAL --> FWD
    TCAL --> GRAPH
    WT --> TCAL
    WT --> GRAPH
    WT --> QGRAPH
    PROF --> QGRAPH
    PROF --> FWD
    PROF --> GRAPH
    QGRAPH --> GRAPH
    QGRAPH --> PLAT
    DEPLOY --> PLAT
    DEPLOY --> GRAPH
```

---

## 2. Pipeline Data Flow

```mermaid
sequenceDiagram
    participant Main as __main__.py
    participant Graph as ONNXGraph
    participant Cali as tensor_calibration
    participant Fwd as forward_net
    participant WT as weight_calibration
    participant QG as quant_graph
    participant Prof as profiling
    participant Deploy as to_deploy

    Main->>Graph: onnx.load() → onnxsim → ONNXGraph 封装

    rect rgb(230,245,255)
        Note over Main,Deploy: Phase 1 — 张量校准
        Main->>Cali: tensor_calibration(onnx_graph, args)
        Cali->>Fwd: 全图 / 子图推理 (ort.InferenceSession)
        Fwd-->>Cali: 逐张量 min/max/optimal_s
        Cali-->>Main: act_clip_val, weight_clip_val
        Main->>Main: save → barrier → reduce → load (跨 rank)
    end

    rect rgb(255,243,230)
        Note over Main,Deploy: Phase 2 — 权重变换 (可选链)
        Main->>WT: weight_calibration(graph, clip_vals, args)
        WT->>WT: bias_correction → WE → update_bn → adaround → brecq → sparse
        WT->>QG: 每步验证: quant_graph 插入 QDQ → 推理 → 对比
        WT-->>Main: graph, graph_ori, updated clip_vals
    end

    rect rgb(230,255,230)
        Note over Main,Deploy: Phase 3 — 量化分析
        Main->>Prof: quantize_profiling_multipass(graph, graph_ori, ...)
        Prof->>QG: quant_graph → insert QDQ nodes
        QG-->>Prof: QDQ 插入后的 ONNX 图
        Prof->>Fwd: FP vs Quant 双路推理
        Fwd-->>Prof: 逐层输出张量
        Prof->>Prof: Cosine Similarity 计算
        Prof-->>Main: layer_cosine_dict, model_cosine_dict
        Main->>Main: save → barrier → reduce → show report
    end

    rect rgb(245,230,255)
        Note over Main,Deploy: Phase 4 — 平台导出
        Main->>Deploy: to_deploy(graph, clip_vals, args)
        Deploy->>Deploy: deploy_dispatcher → gen_*_range/encodings
        Deploy-->>Main: 平台量化参数文件
    end
```

---

## 3. QDQ Graph Construction (核心量化路径)

```mermaid
flowchart TD
    QG["quant_graph()<br/>入口 · 被 9 个函数调用"] --> COPY["graph_q.copy_from(onnx_graph)"]
    COPY --> LOOP{"遍历 quant_node_list<br/>op_type 在 platform_setting 中"}
    LOOP -->|每个 node| INSERT["insert_fake_quant_node()"]

    subgraph Insert["insert_fake_quant_node 内部"]
        direction TB
        I1["遍历 node 的所有输入"] --> I2{"输入类型判断"}
        I2 -->|"is Initializer<br/>且 node 有 weight"| WQ["qw_params<br/>权重量化参数"]
        I2 -->|"第二个 Initializer<br/>有 qb_params"| BQ["qb_params<br/>偏置量化参数"]
        I2 -->|"网络输入 / 中间张量"| AQ["qi_params<br/>激活量化参数"]
        WQ --> GP["get_qnode_by_param()"]
        BQ --> GP
        AQ --> GP
        GP --> SCALE["计算 scale / zero_point<br/>Symmetric: scale = max(|x|)/(2^(bit-1)-1)<br/>Asymmetric: scale = (max-min)/(2^bit-1)"]
        SCALE --> MQD["make_quant_dequant()"]
        MQD --> SUBGRAPH["构建 2 节点 ONNX 子图:<br/>input(fp32) → QuantizeLinear → _q(int8)<br/>→ DequantizeLinear → _dq(fp32)"]
        SUBGRAPH --> INSERT_PURE["graph_q.insert_qnodes_purely(subgraph)"]
    end

    INSERT --> RETOPO["graph_q.topologize_graph()"]
    RETOPO --> LOOP
    LOOP -->|遍历完成| OUT["insert_fake_quant_node_output()<br/>(可选, 量化网络输出)"]
    OUT --> UPDATE["graph_q.update_model()<br/>序列化回 onnx.ModelProto"]
```

---

## 4. ONNXGraph — 中央数据结构

```mermaid
classDiagram
    class ONNXGraph {
        -graph: GraphProto
        -initializer: dict
        -input_map: dict
        -output_map: dict
        -network_inputs: list
        -network_outputs: list
        -tensor_name_shape_map: dict
        -value_name_type_map: dict
        -name_idx_map: dict
        +set_names()
        +convert_constant_to_init()
        +prepare_initializer()
        +get_inp_oup()
        +get_shape_type()
        +get_tensor_shape(name) tuple
        +get_value_type(name) int
        +get_initializer(name) ndarray
        +set_initializer(name, value)
        +get_constant(name) ndarray
        +get_tensor_producer(name) NodeProto
        +get_tensor_consumer(name) list
        +topologize_graph()
        +set_index()
        +index(idx) NodeProto
        +insert_qnodes_purely(subgraph)
        +insert_node_purely(node)
        +remove_node_purely(node)
        +add_network_output(tensor)
        +del_network_output(tensor)
        +del_initializer(name)
        +update_model()
        +update_model_dim()
        +copy_from(other)
        +save_onnx_model(path)
    }

    class QuantizeGraph["quantize.py 调用方"] {
        +quant_graph()
        +insert_fake_quant_node()
        +make_quant_dequant()
        +get_qnode_by_param()
    }

    class Profiling["profiling.py 调用方"] {
        +quantize_profiling_multipass()
        +quantize_profiling_transformer()
        +quantize_profiling_layerwise()
    }

    class WeightTransform["weight_transform/ 调用方"] {
        +weight_calibration()
        +bias_correction()
        +weight_equalization()
        +adaround()
        +brecq()
        +update_bn_multipass()
        +sparse_quant()
    }

    class Deploy["deploy/ 调用方"] {
        +to_deploy()
        +deploy_dispatcher()
        +gen_trt_range()
        +gen_snpe_encodings()
    }

    QuantizeGraph ..> ONNXGraph : uses
    Profiling ..> ONNXGraph : uses
    WeightTransform ..> ONNXGraph : uses
    Deploy ..> ONNXGraph : uses
```

---

## 5. Weight Transform Pipeline

```mermaid
flowchart LR
    INPUT["ONNXGraph<br/>+ act_clip_val<br/>+ weight_clip_val"] --> WT["weight_calibration()"]

    WT --> BC{"--bc ?"}
    BC -->|Yes| BC_DO["bias_correction<br/>修正 Conv/Gemm bias<br/>吸收权重量化误差"]
    BC -->|No| WE{"--we ?"}
    BC_DO --> WE

    WE -->|Yes| WE_DO["weight_equalization<br/>跨层 SConv/SC 缩放<br/>均衡权重分布"]
    WE -->|No| BN{"--update_bn ?"}
    WE_DO --> BN

    BN -->|Yes| BN_DO["update_bn<br/>重算 BN running_mean/var<br/>基于校准数据集"]
    BN -->|No| ADA{"--adaround ?"}
    BN_DO --> ADA

    ADA -->|Yes| ADA_DO["adaround<br/>PyTorch 逐层重建<br/>Adam 优化 soft-rounding V<br/>5000 epochs"]
    ADA -->|No| BRE{"--brecq ?"}
    ADA_DO --> BRE

    BRE -->|Yes| BRE_DO["brecq / QDrop<br/>Block 重建量化<br/>--drop → 激活量化"]
    BRE -->|No| SP{"--sparse ?"}
    BRE_DO --> SP

    SP -->|Yes| SP_DO["sparse_quant<br/>N:M 结构化稀疏"]
    SP -->|No| OUTPUT
    SP_DO --> OUTPUT["更新后的 ONNXGraph<br/>+ 重新校准的 clip_vals"]

    style BC_DO fill:#e3f2fd
    style WE_DO fill:#fff3e0
    style BN_DO fill:#fce4ec
    style ADA_DO fill:#e8f5e9
    style BRE_DO fill:#f3e5f5
    style SP_DO fill:#e0f2f1
```

---

## 6. 分布式执行模型

```mermaid
flowchart TD
    subgraph Init["初始化路径"]
        SLURM["--slurm<br/>init_from_slurm()"]
        MPI["--mpirun<br/>init_from_mpi()"]
        NCCL["默认<br/>torch.distributed<br/>init_process_group(backend='nccl')"]
    end

    subgraph Rank0["Rank 0 (Master)"]
        R0_1["加载 ONNX 模型<br/>创建 ONNXGraph"]
        R0_2["张量校准<br/>(数据均分)"]
        R0_3["reduce_clip_val<br/>汇聚各 rank 结果"]
        R0_4["权重变换<br/>(仅在 Rank 0 执行)"]
        R0_5["reduce_profiling_res<br/>汇聚 profiling 结果"]
        R0_6["show_model_profiling_res<br/>生成量化报告"]
        R0_7["to_deploy<br/>导出平台参数"]
    end

    subgraph RankN["Rank 1..N (Worker)"]
        RN_1["加载 ONNX 模型"]
        RN_2["张量校准<br/>(本 rank 数据分片)"]
        RN_3["save_clip_val<br/>保存本 rank 结果"]
        RN_4["barrier 等待"]
        RN_5["load_clip_val<br/>加载汇聚结果"]
        RN_6["Profiling<br/>(本 rank 数据分片)"]
        RN_7["save_profiling_res<br/>保存本 rank 结果"]
    end

    SLURM & MPI & NCCL --> Init_Done["dist.init 完成<br/>rank · local_rank · world_size"]
    Init_Done --> R0_1 & RN_1
    R0_2 & RN_2 --> R0_3
    R0_3 & RN_3 --> Barrier1["barrier"]
    Barrier1 --> R0_4 & RN_4
    R0_4 & RN_4 --> RN_5
    RN_5 --> R0_5 & RN_6
    R0_5 & RN_6 --> RN_7
    RN_7 --> Barrier2["barrier"]
    Barrier2 --> R0_6
    R0_6 --> R0_7
```

---

## 7. Functional Areas (11 模块)

```mermaid
graph LR
    subgraph Core["核心 (55 symbols · 78% cohesion)"]
        direction TB
        C1["__main__.py · CLI 编排"]
        C2["utils.py · ONNXGraph"]
        C3["forward_net.py · 推理引擎"]
        C4["quantize.py · QDQ 图"]
        C5["profiling.py · 误差分析"]
        C6["platform_settings.py"]
    end

    subgraph WT_Mod["Weight Transform (42 symbols · 86%)"]
        direction TB
        W1["bias_correction.py"]
        W2["weight_equalization.py"]
        W3["update_bn.py"]
        W4["adaround.py"]
        W5["brecq.py"]
        W6["sparse_quant.py"]
    end

    subgraph Deploy_Mod["Deploy (24 symbols · 86%)"]
        direction TB
        D1["TRT · SNPE · QNN"]
        D2["STPU · Atlas · TI"]
        D3["i.MX · RV · MagicMind"]
    end

    subgraph Demos["DemoLab (7 实验套件)"]
        direction TB
        E1["MobileNetV2 + TRT"]
        E2["ResNet18 多后端"]
        E3["OD BEV + QNN"]
        E4["YOLOv8 + SNPE/QNN"]
    end

    Core --> WT_Mod
    Core --> Deploy_Mod
    WT_Mod --> Core
    Demos -.-> Core
    Demos -.-> WT_Mod
    Demos -.-> Deploy_Mod
```

---

## 8. Key Execution Flows

### Flow 1: 主量化分析 (5 steps · 遍历最频繁的路径)

```
quantize_profiling_multipass          dipoorlet/profiling.py
  → quant_graph                       dipoorlet/quantize.py
    → insert_fake_quant_node          dipoorlet/quantize.py
      → get_qnode_by_param            dipoorlet/quantize.py
        → make_quant_dequant          dipoorlet/quantize.py
```

### Flow 2: AdaRound 自适应舍入 (5 steps)

```
adaround                              dipoorlet/weight_transform/adaround.py
  → quant_graph                       dipoorlet/quantize.py
    → insert_fake_quant_node          dipoorlet/quantize.py
      → get_qnode_by_param            dipoorlet/quantize.py
        → make_quant_dequant          dipoorlet/quantize.py
```

### Flow 3: BrecQ / QDrop 块重建 (5 steps)

```
brecq                                 dipoorlet/weight_transform/brecq.py
  → quant_graph                       dipoorlet/quantize.py
    → insert_fake_quant_node          dipoorlet/quantize.py
      → get_qnode_by_param            dipoorlet/quantize.py
        → make_quant_dequant          dipoorlet/quantize.py
```

### Flow 4: 权重校准管道 (4 steps)

```
weight_calibration                    dipoorlet/weight_transform/weight_trans_base.py
  → bias_correction                   dipoorlet/weight_transform/bias_correction.py
    → quant_graph                     dipoorlet/quantize.py
      → copy_from                     dipoorlet/utils.py
```

### Flow 5: 部署导出

```
to_deploy                             dipoorlet/deploy/deploy_base.py
  → deploy_dispatcher                 dipoorlet/deploy/deploy_default.py
    → gen_trt_range / gen_snpe_encodings / gen_stpu_minmax / ...
```

---

## 9. 关键指标汇总

| 符号 | 被多少模块依赖 | 角色 |
|------|:---:|------|
| `ONNXGraph` | 13 | 中央图抽象 — 26 个方法 |
| `quant_graph` | 9 | QDQ 插入汇聚点 |
| `platform_setting_table` | 5+ | 9 后端策略配置 |
| `forward_net` 系列 | 4 | 推理引擎（全图 / 子图） |
| `dispatch_functool` | 3 | 算法 / 平台分派装饰器 |

## 10. 平台覆盖

| 平台 | 模块 | 目标硬件 |
|------|------|----------|
| `trt` | `deploy_trt.py` | NVIDIA GPU (TensorRT) |
| `snpe` | `deploy_snpe.py` | Qualcomm Snapdragon |
| `qnn` | `deploy_qnn.py` | Qualcomm AI Engine |
| `stpu` | `deploy_stpu.py` | Horizon Robotics Journey |
| `atlas` | `deploy_atlas.py` | Huawei Ascend |
| `ti` | `deploy_ti.py` | TI TDA4 automotive |
| `imx` | `deploy_imx.py` | NXP i.MX embedded |
| `rv` | `deploy_rv.py` | Rockchip (RV1126/RK3568) |
| `magicmind` | `deploy_magicmind.py` | Cambricon MLU |

## Tensor Calibration Algorithms

| Algorithm | Flag | Method | Description |
|-----------|------|--------|-------------|
| MSE | `-A mse` (default) | `find_clip_val_octav` | Iterative optimal clipping minimizing MSE |
| MinMax | `-A minmax` | `find_clip_val_minmax` | Simple min/max range across all samples |
| Histogram | `-A hist` | `find_clip_val_hist` | Histogram accumulation with `--threshold` cutoff |
