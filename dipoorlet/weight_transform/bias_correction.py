import numpy as np
from onnx import numpy_helper

from ..forward_net import ActivationCache
from ..quantize import quant_graph
from ..utils import ONNXGraph, logger


def update_conv_node_bias(graph_bc, node, fp_activations, q_activations):
    """
    更新卷积或全连接层节点的偏置（bias）以减少量化误差
    
    参数:
        graph_bc: 用于偏置校正的ONNX图对象
        node: 需要更新偏置的节点
        fp_activations: 全精度（floating point）模型的激活值
        q_activations: 量化（quantized）模型的激活值
    """
    
    # 计算全精度激活值和量化激活值之间的差异
    bias_diff = np.stack(fp_activations, axis=0)  - np.stack(q_activations, axis=0)
    
    # 根据操作类型选择不同的轴进行平均
    # 对于卷积层，在批次(0)和空间维度(2,3)上平均，保留通道维度
    # 对于全连接层(Gemm)，仅在批次维度(0)上平均
    axis = (0, 2, 3) if node.op_type == 'Conv' else (0)
    
    bias_diff = np.squeeze(bias_diff, axis=1).mean(axis=axis)
    
    # 检查节点是否已有偏置参数(通常是input[2])
    if len(node.input) > 2:
        
        # 获取原始偏置值
        ori_bias = numpy_helper.to_array(graph_bc.initializer[node.input[2]][0])
        
        # 将计算出的差异添加到原始偏置上
        corrected_bias = ori_bias + bias_diff
        
        # 获取原始偏置的名称
        corrected_bias_name = graph_bc.initializer[node.input[2]][0].name
        
        # 更新图中的偏置初始化器
        graph_bc.set_initializer(corrected_bias_name, corrected_bias)
        
        # 更新tensor名称到形状的映射，先移除旧名称，再添加新的
        graph_bc.tensor_name_shape_map[corrected_bias_name] = \
            graph_bc.tensor_name_shape_map.pop(graph_bc.initializer[node.input[2]][0].name)
            
        # 将校正后的偏置添加到图的输入列表中
        graph_bc.all_io_input.append(corrected_bias_name)
    else:
        
        # 如果节点没有偏置，就直接使用计算出的差异作为新偏置
        bias = bias_diff
        
        # 为新偏置创建一个名称
        bias_name = node.name + '_bias'
        
        # 将新偏置添加到图中
        graph_bc.set_initializer(bias_name, bias)
        
        # 更新tensor名称到形状的映射
        graph_bc.tensor_name_shape_map[bias_name] = list(bias.shape)
        
        # 将新偏置添加到图的输入列表中
        graph_bc.all_io_input.append(bias_name)
        
        # 查找并更新对应节点的输入列表，添加新偏置
        for bc_node in graph_bc.graph.node:
            if bc_node.name == node.name:
                bc_node.all_io_input.append(bias_name)
                return


def bias_correction(graph, act_clip_val, weight_clip_val, args):
    
    """
    为整个网络图执行偏置校正
    
    通过比较全精度模型和量化模型的激活值差异，计算并应用偏置校正值，
    以减少量化带来的精度损失
    
    参数:
        graph: 原始ONNX图对象
        act_clip_val: 激活值裁剪阈值字典，用于量化激活值
        weight_clip_val: 权重裁剪阈值字典，用于量化权重
        args: 其他参数配置
    """
    
    # 定义需要进行偏置校正的节点类型
    bias_correction_node_type = ['Conv', 'Gemm']
    
    # 合并激活值和权重的裁剪阈值字典
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    
    # 创建一个新的图对象用于偏置校正
    graph_bc = ONNXGraph()
    
    # 从原始图复制所有信息
    graph_bc.copy_from(graph)
    
    # 创建全精度模型的激活缓存，用于获取全精度激活值
    fp_cache = ActivationCache(graph, args)
    
    # 用于存储之前节点的激活缓存，实现增量更新
    prev_act = None
    
    # 遍历原始图中的所有节点
    for node in graph.graph.node:
        
        # 只处理指定类型的节点（卷积和全连接层）
        if node.op_type in bias_correction_node_type:
            logger.info("Update bias for node: {}".format(node.name))
            # We should do incremental update here.
            # 对当前状态的图进行量化，得到量化后的图
            graph_q, _ = quant_graph(graph_bc, clip_val, args)
            
            # 创建量化模型的激活缓存
            q_cache = ActivationCache(graph_q, args)
            
            # 如果有之前的激活缓存，则应用它以实现增量更新
            # 这样可以保持前向计算的一致性
            if prev_act is not None:
                q_cache.activation_cache = prev_act
                
            # 触发对输入节点的前向计算，确保激活缓存被填充
            _ = q_cache[node.input[0]]
            
            # 保存当前的激活缓存，用于下一个节点的处理
            prev_act = q_cache.activation_cache.copy()
            
            # 更新节点的偏置，使用全精度和量化模型的输出差异
            update_conv_node_bias(graph_bc, node, fp_cache[node.output[0]], q_cache[node.output[0]])
            
            # 更新图模型，应用偏置校正
            graph_bc.update_model()
    
    # 保存校正后的模型
    graph_bc.save_onnx_model('update_bias_model')
