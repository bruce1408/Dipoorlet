import copy

import numpy as np
from onnx import numpy_helper

from ..utils import ONNXGraph, logger
from .utils import update_weight

'''
description: 权重均衡化(Weight Equalization)模块
该技术用于量化神经网络时减少精度损失。通过在连续层之间重新分配权重范围，
使得两个连续层的权重分布更加均衡，从而提高量化后模型的性能。
'''

def find_successor(cur_node, graph):
    """
    寻找当前节点的后继节点，支持两种模式：Conv -> Relu -> Conv 或 Conv -> Conv
    
    参数:
        cur_node: 当前节点
        graph: ONNX图对象
    
    返回:
        result: 符合条件的后继节点列表，如果不满足模式条件则返回空列表
    """
    # Conv -> Relu -> Conv or Conv -> Conv pattern supported.
    result = []
    # 获取当前节点的输出张量
    out_tensor = cur_node.output[0]
    # 查找使用该输出张量的节点（消费者）
    nxt_node = graph.get_tensor_consumer(out_tensor)
    for node in nxt_node:
        # 如果消费者是字符串而不是节点对象，返回空列表
        if isinstance(node, str):
            return []
        # 如果下一个节点是Relu或PRelu激活函数
        if node.op_type in ['Relu', 'PRelu']:
            # 获取激活函数的输出
            relu_out = node.output[0]
            # 查找使用该激活函数输出的节点
            nxt_nxt_node = graph.get_tensor_consumer(relu_out)
            for _node in nxt_nxt_node:
                # 如果是卷积层，则添加到结果中
                if not isinstance(_node, str) and _node.op_type == 'Conv':
                    result.append(_node)
                else:
                    # 不符合模式，返回空列表
                    return []
        # 如果下一个节点直接是卷积层
        elif node.op_type == 'Conv':
            result.append(node)
        else:
            # 不符合模式，返回空列表
            return []
    return result


def node_has_equalized(graph, node):
    """
    检查节点是否已经完成了权重均衡化
    
    参数:
        graph: ONNX图对象
        node: 要检查的节点
    
    返回:
        bool: 如果节点有一个符合条件的后继节点，则返回True，否则返回False
    """
    # Helper function for other algos.
    return len(find_successor(node, graph)) == 1


def weight_equalization(graph, args):
    """
    对整个网络执行权重均衡化
    
    参数:
        graph: 原始ONNX图对象
        args: 其他参数配置
    """
    # 创建一个新的图对象用于权重均衡化
    graph_we = ONNXGraph()
    # 从原始图复制所有信息
    graph_we.copy_from(graph)

    # 遍历图中的所有节点
    for node in graph_we.graph.node:
        # 仅处理卷积层
        if node.op_type == 'Conv':
            # 寻找符合条件的后继节点
            succ = find_successor(node, graph_we)
            # 如果没有恰好一个符合条件的后继节点，跳过当前节点
            if len(succ) != 1:
                continue
            # 迭代计数器
            iter = 1
            # 迭代优化权重直到收敛
            while True:
                # 获取第一个卷积层的权重
                weight_first = numpy_helper.to_array(graph_we.initializer[node.input[1]][0])
                # 深拷贝权重以进行修改
                new_weight_first = copy.deepcopy(weight_first)
                # 如果有偏置项，也获取并深拷贝
                if len(node.input) == 3:
                    bias_first = numpy_helper.to_array(graph_we.initializer[node.input[2]][0])
                    new_bias_first = copy.deepcopy(bias_first)
                # 获取后继节点
                next_node = succ[0]
                # 获取第二个卷积层的权重
                weight_second = numpy_helper.to_array(graph_we.initializer[next_node.input[1]][0])
                # 深拷贝权重以进行修改
                new_weight_second = copy.deepcopy(weight_second)
                # 计算分组卷积的组数
                num_group = weight_first.shape[0] // weight_second.shape[1]
                # 记录日志
                logger.info('Cross Layer WE: {} --- {} Groups: {} Iter: {}'.format(node.name, next_node.name, num_group, iter))
                # 计算每组的输入通道数和输出通道数
                group_channels_i = weight_first.shape[0] // num_group
                group_channels_o = weight_second.shape[0] // num_group
                
                # 对每个分组进行权重均衡化
                for g in range(num_group):
                    # 计算当前组在第一个卷积层中的通道范围
                    c_start_i = g * group_channels_i
                    c_end_i = (g + 1) * group_channels_i
                    # 提取第一个卷积层当前组的权重
                    weight_first_group = weight_first[c_start_i:c_end_i]
                    
                    # 计算当前组在第二个卷积层中的通道范围
                    c_start_o = g * group_channels_o
                    c_end_o = (g + 1) * group_channels_o
                    # 提取第二个卷积层当前组的权重
                    weight_second_group = weight_second[c_start_o:c_end_o]
                    
                    # 对每个输入通道进行均衡化
                    for ii in range(weight_second_group.shape[1]):
                        # 计算第一个卷积层对应通道的权重范围（最大绝对值）
                        range_1 = np.abs(weight_first_group)[ii].max()
                        # 计算第二个卷积层与此通道连接的所有权重的范围
                        range_2 = np.abs(weight_second_group)[:, ii].max()
                        
                        # 防止范围过小导致数值不稳定
                        if range_1 < 1e-6:
                            range_1 = 0.
                        if range_2 < 1e-6:
                            range_2 = 0.
                        
                        # 计算缩放因子 s = range_1 / sqrt(range_1 * range_2)
                        # 这个公式使得两层的权重范围几何平均化
                        s = range_1 / np.sqrt(range_1 * range_2)
                        # 处理无穷大或NaN的情况
                        if np.isinf(s) or np.isnan(s):
                            s = 1.0
                        
                        # 应用缩放因子：第一层除以s，第二层乘以s
                        # 这样确保两个层的数学等价性保持不变
                        new_weight_first[c_start_i + ii] /= s
                        new_weight_second[c_start_o:c_end_o, ii] *= s
                        # 如果有偏置项，也需要相应调整
                        if len(node.input) == 3:
                            new_bias_first[c_start_i + ii] /= s

                # 检查是否收敛
                if converged([weight_first, weight_second], [new_weight_first, new_weight_second]):
                    break
                
                # 增加迭代计数
                iter += 1
                
                # 更新图中的权重
                update_weight(graph_we, new_weight_first, node.input[1])
                graph_we.update_model()
                update_weight(graph_we, new_weight_second, next_node.input[1])
                graph_we.update_model()
                # 如果有偏置，也更新偏置
                if len(node.input) == 3:
                    update_weight(graph_we, new_bias_first, node.input[2])
                    graph_we.update_model()
    
    # 保存均衡化后的模型
    graph_we.save_onnx_model('weight_equal_model')


def converged(cur_weight, prev_weight, threshold=1e-4):
    """
    检查权重是否已经收敛
    
    参数:
        cur_weight: 当前权重列表
        prev_weight: 上一次迭代的权重列表
        threshold: 收敛阈值，默认为1e-4
    
    返回:
        bool: 如果前后两次权重的差异小于阈值，则返回True
    """
    # 计算两组权重之间的L2范数（欧氏距离）之和
    norm_sum = 0
    norm_sum += np.linalg.norm(cur_weight[0] - prev_weight[0])
    norm_sum += np.linalg.norm(cur_weight[1] - prev_weight[1])
    # 如果范数和小于阈值，则认为已收敛
    return norm_sum < threshold
