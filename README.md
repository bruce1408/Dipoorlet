# Introduction

Dipoorlet is an offline quantization tool that can perform offline quantization on ONNX model on a given calibration dataset:

* Support several **Activation Calibration** algorithms: ***Mse, Minmax, Hist, etc***.
* Support **Weight Transformation** to achieve better quantization results: ***BiasCorrection, WeightEqualization, etc.***
* Supports **SOTA** offline finetune algorithms to improve quantization performance: ***Adaround, Brecq, Qdrop.***
* Generate **Quantitative Parameters** required for several platforms: ***SNP, TensorRT, STPU, ATLAS, etc.***
* Provide detailed **Quantitative Analysis** to facilitate the identification of accuracy bottlenecks in model quantization.

# Installation

```
git clone https://github.com/ModelTC/Dipoorlet.git
cd Dipoorlet
python setup.py install
```

# Environment

### CUDA

Project using ONNXRuntime as inference runtime, using Pytorch as training tool, so users have to carefully make CUDA and CUDNN version right in order to make this two runtime work.

For example:  
`ONNXRuntime==1.10.0` and `Pytorch==1.10.0-1.13.0` can runs under `CUDA==11.4 CUDNN==8.2.4`

Please visit [ONNXRuntime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) and [Pytorch](https://pytorch.org/get-started/previous-versions/).

### Docker

ONNXRuntime has bug when running in docker when `cpu-sets` is set.
Please check [issue](https://github.com/microsoft/onnxruntime/issues/8313)

# Usage

## Prepare Calibration Dataset

The pre processed calibration data needs to be prepared and provided in a specific path form. For example, the model has two input tensors called "input_0" and "input_1", and the file structure is as follows:

```
cali_data_dir
|
├──input_0
│     ├──0.bin
│     ├──1.bin
│     ├──...
│     └──N-1.bin
└──input_1
      ├──0.bin
      ├──1.bin
      ├──...
      └──N-1.bin
```

## Running Dipoorlet in Pytorch Distributed Environment

```
python -m torch.distributed.launch --use_env -m dipoorlet -M MODEL_PATH -I INPUT_PATH -N PIC_NUM -A [mse, hist, minmax] -D [trt, snpe, rv, atlas, ti, stpu] [--bc] [--adaround] [--brecq] [--drop]
```

## Running Dipoorlet in Cluster Environment

```
python -m dipoorlet -M MODEL_PATH -I INPUT_PATH -N PIC_NUM -A [mse, hist, minmax] -D [trt, snpe, rv, atlas, ti, stpu] [--bc] [--adaround] [--brecq] [--drop] [--slurm | --mpirun]
```

## Optional

* Using -M to specify ONNX model path.
* Using -A to select activation statistic algorithm, minmax, hist, mse.
* Using -D to select deploy platform, trt, snpe, rv, ti...
* Using -N to specify number of calibration pics.
* Using -I to specify path of calibration pics.
* Using -O to specify output path.
* For hist and kl:  
    --bins specify histogram bins.  
    --threshold specify histogram threshold for hist algorithm.
* Using --bc to do Bias Correction algorithm.
* Using --we to do weight equalization.
* Using --adaround to do offline finetune by [Adaround](https://arxiv.org/abs/2004.10568).
* Using --brecq to do offline finetune by [Brecq](https://arxiv.org/abs/2102.05426).
* Using --brecq --drop to do offline finetune by [Qdrop](https://arxiv.org/abs/2203.05740).
* Using --skip_layers to skip quantization of some layers.
* Using --slurm to launch task from slurm.
* Other usage can get by "python -m dipoorlet --h/-help"

## Example

Quantify an onnx model model.onnx, save 100 calibration data in workdir/data/, where "data" represents the name of the onnx model. Use “minmax“ activation value calibration algorithm, use “Qdrop“ to perform unlabeled fine tuning on weights, and finally generate TensorRT quantization configuration information:

##### Calibration Data Path

```
workdir
|
├──data
    ├──0.bin
    ├──1.bin
    ├──...
    └──99.bin

```

##### Command

```
python -m torch.distributed.launch --use_env -m dipoorlet -M model.onnx -I workdir/ -N 100 -A minmax -D trt
```


# Dipoorlet 项目结构

## 技术栈与模型格式
- **技术栈**: ONNX/Onnxruntime，Pytorch（分布式）
- **模型格式**: ONNX

## 框架结构

### 目录结构与功能
- **deploy**  
  包含如何导出每个后端的量化部署配置文件，涉及算法包括 `Minmax`, `Mse`, `Hist`。
  
- **tensor_cali**  
  校准算法相关的文件，涉及 `Minmax`, `Mse`, `Hist`。

- **weight_transform**  
  权重调整算法相关的文件，涉及 `DFQ (we, bc, update_bn)`，以及其他算法如 `AdaRound`, `Brecq`, `Qdrop`。

### Python 文件
- **`__init__.py`**  
  初始化文件。

- **`__main__.py`**  
  主函数文件，包含离线量化的全流程。

- **dist_helper.py**  
  分布式相关的函数。

- **forward_net.py**  
  获取模型输出的一些函数，包含网络输出、层输出和子模型输出。

- **platform_settings.py**  
  配置推理后端的量化设置。

- **profiling.py**  
  用于量化误差分析。

- **quantize.py**  
  包含与量化相关的函数。

- **utils.py**  
  定义了自定义图结构的 `ONNXGraph` 等功能。

## 量化流程
1. **加载ONNX模型**
2. **校准激活/权重生成量化参数**
3. **权重调整**  
   提高量化精度。
4. **量化误差分析**  
   定位量化问题。
5. **输出推理后端的量化配置文件**

## 总结
该框架显示了量化过程的不同环节，包括模型的导入、校准、权重调整、量化误差分析和部署配置的生成，确保量化流程的顺利进行，特别是在分布式和推理优化方面。
