from .basic_algorithm import *

'''
description: 用于对ONNX（Open Neural Network Exchange）模型的张量进行校准
param {*} onnx_graph ONNX模型的图结构，包含了模型的所有层和连接信息
param {*} args       包含校准过程中所需的各种参数和选项，通常是一个命名空间或字典。
return {*}
'''
def tensor_calibration(onnx_graph, args):
    # 使用 minmax 方法找到权重量化参数
    weight_clip_val = find_clip_val_minmax_weight(onnx_graph, args)         
    
    # 使用不同的方法找到激活量化参数
    act_clip_val = tensor_cali_dispatcher(args.act_quant, onnx_graph, args)
    
    return act_clip_val, weight_clip_val
