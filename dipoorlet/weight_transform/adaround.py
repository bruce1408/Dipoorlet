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


'''
description: https://arxiv.org/abs/2006.10518
param {*} graph_ori         原始的graph
param {*} graph             可能经过we bc后的graph
param {*} act_clip_val      激活量化参数
param {*} weight_clip_val   权重量化参数
param {*} args              考虑到激活和权重的量化
return {*}
'''
def adaround(graph_ori, graph, act_clip_val, weight_clip_val, args):
    dist.barrier()
    clip_val = act_clip_val.copy()
    clip_val.update(weight_clip_val)
    graph_ada = copy.deepcopy(graph)
    rank = dist.get_rank()
    num_per_rank = args.data_num // dist.get_world_size()
    rank_st = rank * num_per_rank
    rank_ed = rank_st + num_per_rank
    
    # 比较节省空间的cache, 在进行某层的输入输出的时候, 可以直接从cache中取值, 而不需要重新前向推理, 使用递归的实现
    fp_act_cache = ActivationCache(graph_ori, args, rank_st, rank_ed)
    prev_act_cache = None
    for node in graph_ori.graph.node:
        if node.name in args.skip_layers:
            continue
        if node.op_type in LEARNABLE_LAYER_TYPES:
            # We can not mimic when node has weight equalized.
            if args.we and node_has_equalized(graph, node):
                continue
            if dist.get_rank() == 0:
                logger.info("Adaround for: {}".format(node.name))
            # Using graph_ada and restore act cache for incremental update.
            if not prev_act_cache:
                
                # 给原始graph插入量化反量化节点
                graph_q, quant_node_list = quant_graph(graph_ada, clip_val, args)
                q_act_cache = ActivationCache(graph_q, args, rank_st, rank_ed)
            else:
                q_act_cache.update_graph(graph_q)
                q_act_cache.activation_cache = prev_act_cache
            prev_node = graph_q.get_tensor_consumer(node.input[0])[0]
            prev_node = graph_q.get_tensor_consumer(prev_node.output[0])[0]
            in_tensor_name = node.input[0]
            if prev_node.op_type == QUANT_NODE_NAME_LIST[-1]:
                in_tensor_name = prev_node.output[0]

            q_in_tensor = np.stack(q_act_cache[in_tensor_name])
            fp_out_tensor = np.stack(fp_act_cache[node.output[0]])
            print(node.output[0])
            print(fp_act_cache[node.output[0]].__len__())
            print("the output shape fp out ",fp_out_tensor.shape)
            prev_act_cache = q_act_cache.activation_cache.copy()

            # Get weight and build torch conv.
            weight = numpy_helper.to_array(graph_ada.initializer[node.input[1]][0])
            bias = None
            if len(node.input) == 3:
                bias = numpy_helper.to_array(graph_ada.initializer[node.input[2]][0])

            weight = torch.from_numpy(weight).cuda()
            # Get quantization param.
            if args.deploy != 'nnie':
                weight_range = clip_val[node.input[1]]
                
                # 这个是根据后端相关的，读取量化配置
                qw_param = platform_setting_table[args.deploy]['qw_params']
                if node.op_type == 'ConvTranspose':
                    weight = weight.transpose(0, 1)
                scale, q_min, q_max = get_quant_tensor(weight.shape, qw_param, weight_range)
                
                # 这个就是初始化V的做法
                rest = (weight / scale) - (weight / scale).floor()  
                qw_tensor = {'scale': scale,
                             'q_min': q_min,
                             'q_max': q_max,
                             'per_channel': qw_param['per_channel'],
                             'type': 'Linear'}
            else:
                qw_tensor = {'scale': None,
                             'q_min': None,
                             'q_max': None,
                             'per_channel': None,
                             'type': 'NNIE'}
                rest = nnie_rest_init(weight)
            # Learning. 判断node后面是否有relu这个激活函数，如果有就relu_flag为True，过一下relu激活函数
            relu_flag = follow_relu(graph, node)
            if relu_flag:
                fp_tensor = torch.nn.Parameter(F.relu(torch.from_numpy(fp_out_tensor)), False)
            else:
                fp_tensor = torch.nn.Parameter(torch.from_numpy(fp_out_tensor), False)
            # Learning round mask.
            print("fp tensor.shape", fp_tensor.shape)
            total_iter = args.ada_epoch * np.ceil(num_per_rank / args.ada_bs)
            
            # 根据总的迭代次数来初始化一个正则函数，这个正则就是论文中的F正则
            reg = adaround_reg(total_iter)
            
            # 这个就是优化求解，构建一个torch模型，写一个自定义的qlayer
            # print("node is ", node)
            print(weight.shape) # 32, 3, 3, 3
            print(bias.shape) # 32
            print(rest.shape) # 32, 3, 3, 3
            # print("reg is: ", reg)
            # print(qw_tensor)
            print(relu_flag)    # false
            print(node.op_type) # conv
            print(args.acti_quant) # false
            ada_layer = AdaQLayer(node, weight, bias, rest, reg, qw_tensor, None,
                                  relu_flag, node.op_type, args.acti_quant)
            round_mask = learning_round_mask(
                torch.nn.Parameter(torch.from_numpy(q_in_tensor).cuda(), False),
                fp_tensor.cuda(),
                ada_layer, reg, args.ada_bs, args.ada_epoch)
            # Deploy new weight.
            if args.deploy != 'nnie':
                new_rounded_weight = quant_weight(
                    weight,
                    round_mask, scale, q_min, q_max,
                    qw_param['per_channel'], soft=False)
                if node.op_type == 'ConvTranspose':
                    new_rounded_weight = new_rounded_weight.transpose(0, 1)
            else:
                new_rounded_weight = quant_weight_nnie(weight, round_mask, soft=False)
            new_rounded_weight = new_rounded_weight.cpu().detach().numpy()
            update_weight(graph_ada, new_rounded_weight, node.input[1])
            update_weight(graph_q, new_rounded_weight, node.input[1])
            graph_ada.update_model()
            graph_q.update_model()
    if dist.get_rank() == 0:
        graph_ada.save_onnx_model('adaround')
    # We must use original ranges.
    return graph_ada


def learning_round_mask(in_tensor, fp_out_tensor, ada_layer, reg, batch_size, max_epoch):
    # batch_size = 64 默认
    optimizer = torch.optim.Adam([ada_layer.round_mask])
    ada_layer = DDP(ada_layer, [torch.cuda.current_device()])
    if dist.get_rank() == 0:
        print("in_tensor shape is ========", in_tensor.shape, fp_out_tensor.shape, __file__)
    # New train precedure
    cur_iter = 0
    for epoch in range(1):
        for idx in range(np.ceil(len(in_tensor) / batch_size).astype(int)):
            print("========= the batch is ", np.ceil(len(in_tensor) / batch_size).astype(int), batch_size)
            st = idx * batch_size
            ed = st + batch_size
            print("the st is ", st, __file__)
            print("the ed is ", ed, __file__)
            input = in_tensor[st:ed].squeeze(1)
            fp_output = fp_out_tensor[st:ed].squeeze(1)
            output = ada_layer(input)
            loss = L2_norm(output, fp_output) + reg(ada_layer.module.round_mask, cur_iter)
            cur_iter += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0 and dist.get_rank() == 0:
            logger.info("Epoch: {:<4} L2 Loss: {:>10.3f} Beta: {:>3.3f}".format(epoch, loss, reg.beta))
    res = adaround_reg().rectified_sigmoid(ada_layer.module.round_mask)
    if dist.get_rank() == 0:
        logger.info("Loss: {:>5.3f} Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
            loss,
            res[res + 1e-4 >= 1.0].numel(), res[res <= 1e-4].numel(), torch.numel(res),
            (res[res + 1e-4 >= 1.0].numel() + res[res <= 1e-4].numel()) / torch.numel(res)))
    return ada_layer.module.round_mask
