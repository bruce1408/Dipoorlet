# resnet18 模型量化实验

## fp32
# Accuracy : 70.85%

# =========== trt quant metric ================

## trt int8 native
# Accuracy with TRT int8 infer : 70.75%

## trt int8 dipoorlet mse
# Accuracy with TRT int8 infer : 70.40%

## trt int8 dipoorlet hist
# Accuracy with TRT int8 infer : 70.45%

## trt int8 dipoorlet minmax
# Accuracy with TRT int8 infer : 70.70%

## trt int8 dipoorlet adaround
# Accuracy with TRT int8 infer : 70.60%

## trt int8 dipoorlet brecq
# Accuracy with TRT int8 infer : 70.65%

## trt fp16
# Accuracy with TRT fp16 infer : 70.80%

# =========== qnn quant metric ================

## qnn int8 native
# Accuracy: 69.9%

## qnn dipoorlet mse 
# Accuracy: 36.95%

## qnn dipoorlet hist
# Accuracy: 69.0%

## qnn dipoorlet minmax
# Accuracy: 69.85%

## qnn dipoorlet adaround
# Accuracy: 69.5%

## qnn dipoorlet brecq
# Accuracy: 69.95%

## qnn dipoorlet drop
# Accuracy: 69.65%

## qnn fp16
# Accuracy: 70.8%

# =============================================

## qnn aimet cle 
# Accuracy: 70.6%

## qnn aimet adaround
# Accuracy: 70.25%