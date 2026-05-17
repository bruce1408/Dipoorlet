# Dipoorlet 上手指南

---

## 简介

Dipoorlet 是一个跨边缘设备平台的离线量化工具链，将 FP32 ONNX 模型转换为 INT8，生成适配 9 种推理后端的量化参数配置文件。

---

## 安装

```bash
git clone https://github.com/ModelTC/Dipoorlet.git
cd Dipoorlet
python setup.py install
```

### 环境要求

| 组件 | 版本 |
|------|------|
| CUDA | 11.4 |
| cuDNN | 8.2.4 |
| ONNXRuntime-GPU | 1.10.0 |
| PyTorch | 1.10.0 - 1.13.0 |
| Python | >= 3.6 |

---

## 校准数据准备

```bash
cali_data_dir/
├── input_0/           # 目录名 = ONNX 输入张量名
│   ├── 0.bin          # float32 原始二进制
│   ├── 1.bin
│   └── N-1.bin
└── input_1/
    ├── 0.bin
    └── ...
```

用 `dipoorlet_utils/dataset.py` 生成 `.bin` 文件。

---

## 流水线总览

```
加载ONNX → 校准 → 权重变换 → 误差分析 → 部署

校准: Minmax(快) / Hist(KL截尾) / MSE(OCTAV迭代, 最优)
变换: BiasCorrection → WeightEqualization → UpdateBN → AdaRound/Brecq/Qdrop
分析: 逐层Cosine相似度 / 逐层解除量化敏感度
部署: TRT/SNPE/RV/STPU/Atlas/TI/i.MX/MagicMind/QNN
```

---

## 建议阅读顺序

| 优先级 | 文件 | 作用 |
|--------|------|------|
| 1 | `dipoorlet/__main__.py` | 全流程编排 + CLI参数 |
| 2 | `dipoorlet/utils.py` | `ONNXGraph` 核心数据结构 |
| 3 | `dipoorlet/quantize.py` | Q/DQ量化节点插入逻辑 |
| 4 | `dipoorlet/platform_settings.py` | 各后端量化策略 |
| 5 | `dipoorlet/forward_net.py` | ActivationCache 懒加载推理 |
| 6 | `dipoorlet/weight_transform/weight_trans_base.py` | 权重变换调度 |

---

## 快速开始

```bash
# 基线量化
torchrun --nproc_per_node=1 -m dipoorlet \
    -M model.onnx -I calib_data/ -N 128 -A minmax -D trt

# 最佳性价比 (大多数CNN)
torchrun --nproc_per_node=1 -m dipoorlet \
    -M model.onnx -I calib_data/ -N 128 -A mse -D trt --adaround

# 最大精度
torchrun --nproc_per_node=1 -m dipoorlet \
    -M model.onnx -I calib_data/ -N 128 -A mse -D trt --brecq --drop
```

---

## 算法选择指南

| 场景 | 推荐参数 |
|------|----------|
| 快速基线 | `-A minmax` |
| 较优基线 | `-A mse` |
| 轻度精度提升 (~1-2%) | `--adaround` |
| 中度精度恢复 | `--bc --we --adaround` |
| 最大精度(慢) | `--brecq --drop` |

---

## 各平台量化策略差异

| 平台 | 权重 | 激活 | 特殊 |
|------|------|------|------|
| TRT | 对称 INT8, per_channel | 对称 INT8 | 基础算子 |
| SNPE | 非对称 INT8, per_channel | 非对称 INT8 | 全量+Sigmoid |
| RV | 非对称, per_tensor | 非对称 | 量化输出 |
| Atlas | 对称 INT8, per_channel | 非对称 INT8 | Conv/Gemm/AvgPool |
| TI | 对称 INT8, per_tensor | 对称, log_scale | dynamic_sym |
| i.MX | 对称 INT8, per_channel, log_scale | 对称, log_scale | 量化输出 |
| MagicMind | 非对称 INT8, per_channel | 非对称 INT8 | Gemm/Conv/MatMul |

---

## 特殊模式

### 跳过敏感层
```bash
--skip_layers "/layer1/conv1" "/layer2/conv2"
```

### 逐层敏感度分析
```bash
--layerwise_error_prof --prof_num 32
```

### 分布式
```bash
torchrun --nproc_per_node=4 -m dipoorlet ...
# 或
--slurm / --mpirun
```

---

## 新增后端 (3步)

1. 在 `dipoorlet/platform_settings.py` 添加平台配置
2. 在 `dipoorlet/deploy/deploy_xxx.py` 中用 `@deploy_register.register('xxx')` 注册
3. 在 `dipoorlet/deploy/__init__.py` 中 import

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `act_clip_val.json` | 激活量化范围 |
| `weight_clip_val.json` | 权重量化范围(per_channel) |
| `quant_model.onnx` | 插入Q/DQ的量化模型 |
| `trt_clip_val.json` | TensorRT 配置 |
| `snpe_encodings.json` | SNPE 编码配置 |
| `rv_quantized_param.json` | 瑞芯微 配置 |
| `layer_res.json.rank0` | 逐层量化误差 |
| `log-*.txt` | 详细运行日志 |

所有输出默认在 `{model_path}/results/`，通过 `-O` 自定义。

---

## 核心设计模式

### 装饰器注册表 (用于算法分发和后端路由)

```python
class DeployRegister:
    def __init__(self):
        self._deploy_dict = {}
    def register(self, key):
        def wrapper(func):
            self._deploy_dict[key] = func
            return func
        return wrapper
```

### ActivationCache (大模型显存优化)

懒加载 + 引用计数：每个节点拆成独立子图，按需推理，引用归零时立即释放内存。
