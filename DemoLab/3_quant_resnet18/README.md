# resnet18 模型量化实验

## FP32 Baseline

- **Accuracy**: 70.85%

## TensorRT (TRT) 量化指标

| 量化方法          | 准确率 (Accuracy) |
| ----------------- | ------------------- |
| TRT Native INT8   | 70.75%              |
| TRT Native FP16   | 70.80%              |
| Dipoorlet MSE     | 70.40%              |
| Dipoorlet Hist    | 70.45%              |
| Dipoorlet MinMax  | 70.70%              |
| Dipoorlet AdaRound| 70.60%              |
| Dipoorlet BRECQ   | 70.65%              |


## QNN 量化指标

| 量化方法            | 准确率 (Accuracy) |
| ------------------- | ------------------- |
| QNN Native INT8     | 69.9%               |
| QNN Native FP16     | 70.8%               |
| Dipoorlet MSE       | 36.95%              |
| Dipoorlet Hist      | 69.0%               |
| Dipoorlet MinMax    | 69.85%              |
| Dipoorlet AdaRound  | 69.5%               |
| Dipoorlet BRECQ     | 69.95%              |
| Dipoorlet Drop      | 69.65%              |
| AIMET CLE           | 70.6%               |
| AIMET AdaRound      | 70.25%              |
