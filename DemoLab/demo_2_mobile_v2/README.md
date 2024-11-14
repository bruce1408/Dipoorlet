# mobile_v2 模型量化实验

## float

Accuracy with onnx fp16 infer : 69.05%

## dipoorlet int8 act->mse brecq

Accuracy with Dipoorlet int8 infer : 68.55%

## dipoorlet int8 act->mse

Accuracy with Dipoorlet int8 infer : 68.12%

## dipoorlet int8 act->hist

Accuracy with TRT int8 infer : 66.989%

## dipoorlet int8 act->minmax

Accuracy with TRT int8 infer : 64.97%

## trt int8 kl散度

Accuracy with TRT int8 infer : 66.13%
