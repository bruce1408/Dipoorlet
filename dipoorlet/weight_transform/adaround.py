import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from onnx import numpy_helper
from torch.nn.parallel import DistributedDataParallel as DDP
from ..forward_net import ActivationCache
from ..platform_settings import platform_setting_table
from ..quantize import QUANT_NODE_NAME_LIST, quant_graph
from ..utils import logger
from .ada_quant_layer import *
from .utils import *
from .weight_equalization import node_has_equalized
from quant_tools.common_utils import *


'''
description: 基于论文 https://arxiv.org/abs/2006.10518 实现的AdaRound方法
AdaRound是一种用于优化量化神经网络的技术，通过学习权重的舍入策略来减少量化误差

参数:
    graph_ori: 原始的ONNX图模型
    graph: 可能经过权重均衡化(we)或偏置校正(bc)后的ONNX图模型
    act_clip_val: 激活值量化参数，用于裁剪激活值的范围
    weight_clip_val: 权重量化参数，用于裁剪权重的范围
    args: 配置参数，包括量化位宽、数据集大小等
返回:
    graph_ada: 应用AdaRound后的ONNX图模型
'''

@time_it  # 装饰器用于计时函数执行时间
def adaround(graph_ori, graph, act_clip_val, weight_clip_val, args):
    
    # dist.barrier()  # 分布式训练时的同步点，确保所有进程同步执行
    
    # 合并激活值和权重的裁剪参数
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    
    # 创建图的深拷贝，用于AdaRound处理
    graph_ada = copy.deepcopy(graph)
    
    # 分布式训练相关设置（已注释）
    # rank = dist.get_rank()
    # num_per_rank = args.data_num // dist.get_world_size()
    # rank_st = rank * num_per_rank
    # rank_ed = rank_st + num_per_rank
    
    # 非分布式训练时使用全部数据
    rank_st = 0
    rank_ed = args.data_num
    
    # 创建原始精度模型的激活缓存，用于存储各层的输入输出
    # 激活缓存可以节省空间，避免重复前向计算
    fp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    
    # 用于存储前一层的激活缓存，实现增量更新
    prev_act_cache = None
    
    # 遍历原始图中的所有节点
    for node in graph_ori.graph.node:
        # 跳过用户指定的不需要处理的层
        if node.name in args.skip_layers:
            continue
            
        # 只处理可学习的层（如卷积、全连接等）
        if node.op_type in LEARNABLE_LAYER_TYPES:
            # 如果启用了权重均衡化且节点已进行过均衡化，则跳过
            if args.we and node_has_equalized(graph, node):
                continue
                
            # 打印日志信息
            # if dist.get_rank() == 0:
            logger.info("Adaround for: {}".format(node.name))
            
            # 如果没有前一层的激活缓存，说明是第一次处理
            if not prev_act_cache:
                # 给当前图插入量化和反量化节点
                graph_q, quant_node_list = quant_graph(graph_ada, clip_val, args)
                # 创建量化模型的激活缓存
                q_act_cache = ActivationCache(graph_q, args, rank_st, rank_ed)
            else:
                # 使用增量更新方式：更新图并复用之前的激活缓存
                q_act_cache.update_graph(graph_q)
                q_act_cache.activation_cache = prev_act_cache
                
            # 获取当前节点的前一个节点（通常是量化节点）
            prev_node = graph_q.get_tensor_consumer(node.input[0])[0]
            prev_node = graph_q.get_tensor_consumer(prev_node.output[0])[0]
            
            # 获取输入张量名称
            in_tensor_name = node.input[0]
            if prev_node.op_type == QUANT_NODE_NAME_LIST[-1]:  # 如果前一个节点是反量化节点
                in_tensor_name = prev_node.output[0]
                
            # 获取量化模型的输入张量，堆叠为一个批次
            # 形状例如：1,1,3,720,1920
            q_in_tensor = np.stack(q_act_cache[in_tensor_name])
            
            # 获取全精度模型的输出张量，用于计算误差
            fp_out_tensor = np.stack(fp_act_cache[node.output[0]])
           
            # 保存当前的激活缓存，用于下一层处理
            prev_act_cache = q_act_cache.activation_cache.copy()

            # 获取节点的权重参数
            weight = numpy_helper.to_array(graph_ada.initializer[node.input[1]][0])
            
            # 获取偏置参数（如果有）
            bias = None
            if len(node.input) == 3:
                bias = numpy_helper.to_array(graph_ada.initializer[node.input[2]][0])

            # 将权重转换为PyTorch张量并移至GPU
            weight = torch.from_numpy(weight).cuda()
            
            # 获取量化参数
            if args.deploy != 'nnie':  # 根据部署平台选择不同的量化策略
                # 获取权重的量化范围
                weight_range = clip_val[node.input[1]]
                
                # 根据部署平台获取权重量化参数配置
                qw_param = platform_setting_table[args.deploy]['qw_params']
                
                # 对于转置卷积，需要转置权重
                if node.op_type == 'ConvTranspose':
                    weight = weight.transpose(0, 1)
                    
                # 计算量化参数：缩放因子和量化范围
                scale, q_min, q_max = get_quant_tensor(weight.shape, qw_param, weight_range)
                
                # 计算每个权重值的小数部分，用于初始化AdaRound的V值
                # 论文中的公式：V = W/s - floor(W/s)
                rest = (weight / scale) - (weight / scale).floor()
                
                # 设置权重量化张量的参数
                qw_tensor = {
                    'scale': scale,
                    'q_min': q_min,
                    'q_max': q_max,
                    'per_channel': qw_param['per_channel'],
                    'type': 'Linear'
                }
            else:  # 针对NNIE平台的特殊处理
                qw_tensor = {
                    'scale': None,
                    'q_min': None,
                    'q_max': None,
                    'per_channel': None,
                    'type': 'NNIE'
                }
                
                # 使用NNIE特定的方法初始化rest值
                rest = nnie_rest_init(weight)
            
            # 检查当前节点后面是否有ReLU激活函数
            relu_flag = follow_relu(graph, node)
            
            # 根据是否有ReLU激活函数，处理全精度模型的输出
            if relu_flag:
                fp_tensor = torch.nn.Parameter(F.relu(torch.from_numpy(fp_out_tensor)), False)
            else:
                fp_tensor = torch.nn.Parameter(torch.from_numpy(fp_out_tensor), False)
            
            # 计算总迭代次数：轮数 * 向上取整(数据量/批次大小)
            total_iter = args.ada_epoch * np.ceil(args.data_num / args.ada_bs)
            
            # 初始化AdaRound的正则化函数，根据总迭代次数设置beta的退火策略
            # 这个正则项用于促使舍入掩码值趋近于0或1
            reg = adaround_reg(total_iter)
            
            # 调试模式下打印更多信息
            if args.debug_dipoorlet:
                print(node.output[0])
                print(fp_act_cache[node.output[0]].__len__())
                print("the output shape fp out ",fp_out_tensor.shape)
                print("fp tensor.shape", fp_tensor.shape)
                print(weight.shape)     # 例如：32, 3, 3, 3（输出通道，输入通道，核高，核宽）
                print(bias.shape)       # 例如：32（输出通道数）
                print(rest.shape)       # 与权重形状相同：32, 3, 3, 3
                print(relu_flag)        # 是否有ReLU：true/false
                print(node.op_type)     # 操作类型：如Conv
                print(args.acti_quant)  # 是否量化激活值：true/false
            
            # 创建AdaRound量化层，用于优化舍入策略
            # 这个层实现了论文中的可微分量化过程
            ada_layer = AdaQLayer(node, weight, bias, rest, reg, qw_tensor, None, relu_flag, node.op_type, args.acti_quant)
            
            # 学习舍入掩码：优化权重的舍入方式，使量化误差最小
            round_mask = learning_round_mask(
                torch.nn.Parameter(torch.from_numpy(q_in_tensor).cuda(), False),  # 量化模型输入
                fp_tensor.cuda(),                                                 # 全精度模型输出
                ada_layer,                                                       # AdaRound层
                reg,                                                             # 正则化函数
                args.ada_bs,                                                     # 批次大小
                args.ada_epoch,                                                  # 训练轮数
                args.debug_dipoorlet)                                            # 是否调试模式
            
            # 应用学习到的舍入策略，得到新的量化权重
            if args.deploy != 'nnie':
                # 使用学习到的掩码对权重进行量化（硬量化，soft=False）
                new_rounded_weight = quant_weight(
                    weight,
                    round_mask, scale, q_min, q_max,
                    qw_param['per_channel'], soft=False)
                    
                # 对于转置卷积，需要再次转置回去
                if node.op_type == 'ConvTranspose':
                    new_rounded_weight = new_rounded_weight.transpose(0, 1)
            else:
                # NNIE平台特定的权重量化方法
                new_rounded_weight = quant_weight_nnie(weight, round_mask, soft=False)
                
            # 将结果转回CPU并转为NumPy数组
            new_rounded_weight = new_rounded_weight.cpu().detach().numpy()
            
            # 更新两个图中的权重
            update_weight(graph_ada, new_rounded_weight, node.input[1])
            update_weight(graph_q, new_rounded_weight, node.input[1])
            
            # 更新模型，应用新权重
            graph_ada.update_model()
            graph_q.update_model()
    
    # 保存AdaRound处理后的模型
    # if dist.get_rank() == 0:
    graph_ada.save_onnx_model('adaround')
    
    # 返回处理后的图（使用原始的量化范围）
    return graph_ada


def learning_round_mask(in_tensor, fp_out_tensor, ada_layer, reg, batch_size, max_epoch, debug):
    """
    学习权重的最优舍入策略（舍入掩码）
    
    参数:
        in_tensor: 量化模型的输入张量
        fp_out_tensor: 全精度模型的输出张量（目标）
        ada_layer: AdaRound量化层
        reg: 正则化函数，用于促使舍入掩码二值化
        batch_size: 批次大小
        max_epoch: 最大训练轮数
        debug: 是否启用调试模式
    
    返回:
        round_mask: 学习到的舍入掩码，用于确定每个权重是向上还是向下舍入
    """
    # 使用Adam优化器优化舍入掩码
    optimizer = torch.optim.Adam([ada_layer.round_mask])
    
    # 分布式训练设置（已注释）
    # ada_layer = DDP(ada_layer, [torch.cuda.current_device()])
    # if dist.get_rank() == 0:
    
    # 开始训练循环
    cur_iter = 0
    for epoch in range(1):  # 只运行一个周期
        # 计算需要的批次数量
        for idx in range(np.ceil(len(in_tensor) / batch_size).astype(int)):
            # 计算当前批次的起始和结束索引
            st = idx * batch_size
            ed = st + batch_size
           
            # 获取当前批次的输入和目标输出
            input = in_tensor[st:ed].squeeze(1)  # 移除维度为1的轴
            fp_output = fp_out_tensor[st:ed].squeeze(1)
            
            # 通过AdaRound层前向传播，得到量化后的输出
            output = ada_layer(input)
            
            # 调试模式下打印形状信息
            if debug:
                print("the st is ", st)     # 批次起始索引，如0
                print("the ed is ", ed)     # 批次结束索引，如64
                print("Input shape: ", input.shape)  # 输入形状，如[1, 3, 720, 1920]
                print("FP Output shape:", fp_output.shape)  # 全精度输出形状，如[1, 32, 368, 960]
                print("the output is ", output.shape)  # AdaRound输出形状，如[1, 32, 360, 960]
                print("the fp_output is ", fp_output.shape)  # 重复打印全精度输出
            
            # 计算损失：L2范数（输出误差）+ 正则项（促使舍入掩码二值化）
            loss = L2_norm(output, fp_output) + reg(ada_layer.round_mask, cur_iter)
            
            # 更新迭代计数
            cur_iter += 1
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # 每50轮打印一次损失值和beta参数
        # if epoch % 50 == 0 and dist.get_rank() == 0:
        if epoch % 50 == 0:
            logger.info("Epoch: {:<4} L2 Loss: {:>10.3f} Beta: {:>3.3f}".format(epoch, loss, reg.beta))
    
    # 应用修正后的sigmoid函数，将舍入掩码转换为接近0或1的值
    res = adaround_reg().rectified_sigmoid(ada_layer.round_mask)
    
    # 打印最终结果统计
    # if dist.get_rank() == 0:
    if True:
        logger.info("Loss: {:>5.3f} Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
            loss,
            res[res + 1e-4 >= 1.0].numel(),  # 向上舍入的权重数量
            res[res <= 1e-4].numel(),        # 向下舍入的权重数量
            torch.numel(res),                # 总权重数量
            (res[res + 1e-4 >= 1.0].numel() + res[res <= 1e-4].numel()) / torch.numel(res)))  # 已确定舍入方向的比例
    
    # 返回学习到的舍入掩码
    return ada_layer.round_mask