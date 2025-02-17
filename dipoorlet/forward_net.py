import copy
import time
import sys, os
from collections import OrderedDict

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.distributed as dist
from onnx.helper import make_graph, make_model
from onnx.helper import make_tensor_value_info as mtvi
from tqdm import tqdm

from .platform_settings import platform_setting_table
from .quantize import QUANT_NODE_NAME_LIST
from .utils import ONNXGraph, logger

# 设置默认的日志级别为3（警告）
ort.set_default_logger_severity(3)

# 设置递归深度限制
sys.setrecursionlimit(2000)

class ActivationCache(object):
    # 假设通过序列获取张量
    # We assume get tensor by sequence.
    def __init__(self, graph, args, st=None, ed=None):
        # 深拷贝图对象，防止修改原始图
        self.graph = copy.deepcopy(graph)
        self.graph_list = []        # 存储子图的列表
        self.ref_cnt = {}           # 引用计数器
        self.name_to_net = {}       # 名称到网络的映射
        self.name_to_graph_id = {}  # 名称到图ID的映射
        self.activation_cache = {}  # 激活缓存
        self.args = args            # 参数
        self.st = st                # 开始索引
        self.ed = ed                # 结束索引
        self.debug = True
        # 设置CUDA执行提供者
        self.providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
        self.fetch_input()          # 获取模型的输入数据，并reshape成网络输入的形状
        self._split_network()       # 分割网络
        self.fill_ref_cnt()         # 填充引用计数

    def reset(self):
        # 清空激活缓存并重新获取输入和引用计数
        self.activation_cache.clear()
        self.fetch_input()
        self.fill_ref_cnt()

    def fetch_input(self, in_tensor=None):
        # 如果没有指定张量，初始化时获取所有输入
        if in_tensor is None:
            # Means We are initializing.
            for name in self.graph.network_inputs:
                self.activation_cache[name] = []
            if self.st is None:
                self.st = 0
                self.ed = self.args.data_num
            # 生成输入数据
            for data in input_data_generator(self.args.input_dir, self.graph.network_inputs, self.st, self.ed):
                for name in self.graph.network_inputs:
                    self.activation_cache[name].append(
                        data[name][:].reshape(*self.graph.get_tensor_shape(name)).copy())
        else:
            # 如果指定了张量，获取特定张量的数据.
            self.activation_cache[in_tensor] = []
            for data in input_data_generator(self.args.input_dir, self.graph.network_inputs, self.st, self.ed):
                self.activation_cache[in_tensor].append(
                    data[in_tensor][:].reshape(*self.graph.get_tensor_shape(in_tensor)).copy())

    def input_generator(self, tensor_name_list):
        # TODO batch generator.
        data = {}
        for i in range(self.ed - self.st):
            for tensor in tensor_name_list:
                data[tensor] = self.activation_cache[tensor][i]
            yield data

    def __getitem__(self, tensor_name):
        # 获取张量的值
        if tensor_name in self.graph.initializer:
            return self.graph.initializer[tensor_name][0]
        if tensor_name not in self.activation_cache:
            node = self.graph.get_tensor_producer(tensor_name)
            # quantize_output(self.name, 'get item: ', tensor_name, self.activation_cache.keys())
            self.forward_subnet(node.name, node.input)
        return self.activation_cache[tensor_name]

    def forward_subnet(self, subnet_name, input_list):
        # 前向传播子网络
        sub_graph = self.graph_list[self.name_to_graph_id[subnet_name]]
        for input_tensor in input_list:
            if input_tensor == '':
                continue
            if input_tensor not in sub_graph.initializer and input_tensor not in self.activation_cache:
                node = self.graph.get_tensor_producer(input_tensor)
                if isinstance(node, str):
                    # Means We need network input.
                    self.fetch_input(input_tensor)
                else:
                    self.forward_subnet(node.name, node.input)

        input_generator = self.input_generator(sub_graph.network_inputs)
        sub_graph = self.graph_list[self.name_to_graph_id[subnet_name]]
        sub_net = sub_graph.model
        ort_inputs = {}
        ort_session = ort.InferenceSession(sub_net.SerializeToString(), providers=self.providers)
        if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
            logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")

        for data in input_generator:
            for name in sub_graph.network_inputs:
                if len(data[name].shape) == 0 or sub_graph.get_tensor_shape(name)[0] == 0:
                    ort_inputs[name] = data[name]
                else:
                    ort_inputs[name] = data[name][:].reshape(*sub_graph.get_tensor_shape(name))
            outputs = [output.name for output in ort_session.get_outputs()]
            ort_outputs = ort_session.run(outputs, ort_inputs)
            ort_outs = OrderedDict(zip(outputs, ort_outputs))

            for i in ort_outs:
                # There may be dummy outputs, which
                # do not needed by any other layers neither is network output.
                if i in self.ref_cnt or i in self.graph.network_outputs:
                    if i in self.activation_cache:
                        self.activation_cache[i].append(ort_outs[i].copy())
                    else:
                        self.activation_cache[i] = [ort_outs[i].copy()]
        # Tensor Wont Be used in this forward.
        for input_tensor in input_list:
            if input_tensor in sub_graph.initializer:
                continue
            if input_tensor == '':
                continue
            self.ref_cnt[input_tensor] -= 1
            if self.ref_cnt[input_tensor] == 0:
                del (self.activation_cache[input_tensor])

    def fill_ref_cnt(self):
        # 填充引用计数
        for node in self.graph.graph.node:
            for in_tensor in node.input:
                if in_tensor in self.ref_cnt:
                    self.ref_cnt[in_tensor] += 1
                else:
                    self.ref_cnt[in_tensor] = 1

    def _split_network(self):
        # 分割网络为子图
        for i, node in enumerate(self.graph.graph.node):
            inputs = []  # 存储子图的输入
            outputs = []  # 存储子图的输出
            inits = []  # 存储子图的初始化器
            network_inputs = []  # 存储网络输入
            network_outputs = []  # 存储网络输出
            
            # 处理节点的输入
            for input in node.input:
                if input == '':
                    continue
                if input not in self.graph.initializer:
                    in_type = self.graph.get_value_type(input)
                    shape = self.graph.get_tensor_shape(input)
                    if shape[0] == 0:
                        shape = []
                    input_value = mtvi(input, in_type, shape)
                    inputs.append(input_value)
                    network_inputs.append(input)
                else:
                    inits.append(self.graph.initializer[input][0])
            
            # 处理节点的输出
            for output in node.output:
                if output == '':
                    continue
                out_type = self.graph.get_value_type(output)
                shape = self.graph.get_tensor_shape(output)
                if shape[0] == 0:
                    shape = []
                output_value = mtvi(output, out_type, shape)
                outputs.append(output_value)
                network_outputs.append(output)

            # 创建子图
            graph = make_graph(nodes=[node], name=node.name, inputs=inputs,
                               outputs=outputs, initializer=inits)
                
            opset_import = self.graph.model.opset_import
            sub_net = make_model(graph, producer_name=node.name, opset_imports=opset_import)
            sub_graph = ONNXGraph(sub_net, self.args.output_dir)
            
            if self.debug:
                # os.makedirs(self.args.output_debug_dir, exist_ok=True)
                onnx.save(sub_net, f"/mnt/share_disk/bruce_trie/onnx_models/dipoorlet_debug_onnx_models/{node.name}.onnx")

            sub_graph.tensor_name_shape_map = self.graph.tensor_name_shape_map
            sub_graph.network_inputs = network_inputs
            sub_graph.network_outputs = network_outputs
            self.graph_list.append(sub_graph)
        
        # 更新图的id映射
        for idx, sub_graph in enumerate(self.graph_list):
            self.name_to_graph_id[sub_graph.graph.name] = idx

    def update_graph(self, graph):
        # 更新图
        for i, sub_graph in enumerate(self.graph_list):
            for init_name in self.graph_list[i].initializer:
                tensor = graph.get_initializer(init_name)
                self.graph_list[i].set_initializer(init_name, tensor)
            self.graph_list[i].update_model()
        self.ref_cnt = {}
        self.fill_ref_cnt()


def forward_get_minmax(onnx_graph, args):
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")
    # Start activation quantization.
    statistics = {}
    t1 = 0
    ort_inputs = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    for data in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                     desc='Minmax update'):
        for name in onnx_graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(onnx_graph.get_tensor_shape(name))
        st = time.time()
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ed = time.time()
        t1 += ed - st
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
        for i in ort_inputs:
            if i in statistics:
                statistics[i]['max'].append(ort_inputs[i].max())
                statistics[i]['min'].append(ort_inputs[i].min())
            else:
                statistics[i] = {}
                statistics[i]['max'] = [ort_inputs[i].max()]
                statistics[i]['min'] = [ort_inputs[i].min()]
        for i in ort_outs:
            if i in statistics:
                statistics[i]['max'].append(ort_outs[i].max())
                statistics[i]['min'].append(ort_outs[i].min())
            else:
                statistics[i] = {}
                statistics[i]['max'] = [ort_outs[i].max()]
                statistics[i]['min'] = [ort_outs[i].min()]
    logger.info("Forward time: {:.2f} seconds".format(t1))
    return statistics


def forward_get_hist(onnx_graph, stats_min_max, args):
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA may not used. Please check your ort/cuda/cudnn version.")
    # Start activation quantization.
    statistics = {}
    ort_inputs = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    for data in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                     desc='Hist update: {}'.format(args.rank)):
        for name in onnx_graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(onnx_graph.get_tensor_shape(name))
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ort_outs = OrderedDict(zip(outputs, ort_outputs))

        for i in ort_inputs:
            data_max = max(np.max(stats_min_max[i]['max']),
                           -np.min(stats_min_max[i]['min']))
            hist, _ = np.histogram(np.abs(ort_inputs[i]), int(args.bins), (0, data_max))
            if i in statistics:
                statistics[i].append(hist)
            else:
                statistics[i] = [hist]
        for i in ort_outs:
            data_max = max(np.max(stats_min_max[i]['max']),
                           -np.min(stats_min_max[i]['min']))
            hist, _ = np.histogram(np.abs(ort_outs[i]), int(args.bins), (0, data_max))
            if i in statistics:
                statistics[i].append(hist)
            else:
                statistics[i] = [hist]
    return statistics

def forward_net_octav(onnx_graph, args):
    # 生成图和网络的深拷贝，防止对原始图的修改
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    
    # 确保所有节点的输出都在图的输出列表中
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in [_o.name for _o in graph.output]:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    
    # 设置CUDA执行提供者
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    
    # 检查CUDA是否被使用
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA可能未被使用。请检查您的ort/cuda/cudnn版本。")

    # 初始化统计信息字典和计时器
    statistics = {}
    t1 = 0
    
    # 计算每个进程处理的数据范围
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    
    # 获取ORT会话的输出名称
    outputs = [output.name for output in ort_session.get_outputs()]
    
    # 遍历数据批次，进行前向传播和统计
    for data_batch in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
                           desc='OCTAV更新 rank: {}'.format(args.rank)):
        
        # 准备ORT输入
        ort_inputs = {name: data_batch[name][:].reshape(onnx_graph.get_tensor_shape(name)) 
                      for name in onnx_graph.network_inputs}
        
        st = time.time()
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ed = time.time()
        t1 += ed - st

        # 将输出结果与输入合并
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
        ort_inputs.update(ort_outs)

        # 计算每个张量的统计信息
        for i, tensor in ort_inputs.items():
            data_max = np.max(tensor)
            data_min = np.min(tensor)
            
            # 判断是否使用动态对称量化 如果dynamic_sym = True，意味着多一位
            unsigned = 4 if (np.abs(data_min) < 1e-6 and 
                             'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']) else 1

            abs_x = np.abs(tensor)
            non_zero_mask = abs_x > 0
            s_n = abs_x.sum() / np.count_nonzero(non_zero_mask)

            # 迭代计算最优的s_n
            for _ in range(20):
                mask = abs_x > s_n
                s_n_plus_1 = abs_x[mask].sum() / (1 / (4 ** 8) / 3 / unsigned * np.sum(~mask) + np.sum(mask))
                if np.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1

            # 更新统计信息
            if i in statistics:
                statistics[i]['optimal_s'].append(s_n)
                statistics[i]['min'].append(data_min)
                statistics[i]['max'].append(data_max)
            else:
                statistics[i] = {
                    'optimal_s': [s_n],
                    'min': [data_min],
                    'max': [data_max]
                }
    # 记录并输出前向传播的总时间
    logger.info("前向传播时间: {:.2f} 秒".format(t1))
    return statistics

def forward_get_minmax_transformer(onnx_graph, args):
    # Start minmax activation quantization.
    statistics = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    fp_act_cache = ActivationCache(onnx_graph, args, data_st_idx, data_ed_idx)

    input_names = [inp.name for inp in onnx_graph.graph.input]
    output_names = []
    for node in onnx_graph.graph.node:
        for out in node.output:
            output_names.append(out)
    tensor_names = input_names + output_names

    st = time.time()
    for name in tensor_names:
        if name == '':
            continue
        for i in range(data_ed_idx - data_st_idx):
            if name in statistics:
                statistics[name]['max'].append(fp_act_cache[name][i].max())
                statistics[name]['min'].append(fp_act_cache[name][i].min())
            else:
                statistics[name] = {}
                statistics[name]['max'] = [fp_act_cache[name][i].max()]
                statistics[name]['min'] = [fp_act_cache[name][i].min()]
    ed = time.time()

    logger.info("Forward time: {:.2f} seconds".format(ed - st))
    return statistics


def forward_get_hist_transformer(onnx_graph, stats_min_max, args):
    # Start hist activation quantization.
    statistics = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    fp_act_cache = ActivationCache(onnx_graph, args, data_st_idx, data_ed_idx)

    input_names = [inp.name for inp in onnx_graph.graph.input]
    output_names = []
    for node in onnx_graph.graph.node:
        for out in node.output:
            output_names.append(out)
    tensor_names = input_names + output_names

    for name in tensor_names:
        if name == '':
            continue
        for i in range(data_ed_idx - data_st_idx):
            data_max = max(np.max(stats_min_max[name]['max']),
                           -np.min(stats_min_max[name]['min']))
            hist, _ = np.histogram(np.abs(fp_act_cache[name][i]), int(args.bins), (0, data_max))
            if name in statistics:
                statistics[name].append(hist)
            else:
                statistics[name] = [hist]

    return statistics


def forward_net_octav_transformer(onnx_graph, args):
    # Start mse activation quantization.
    statistics = {}
    rank_num = args.data_num // args.world_size
    data_st_idx = args.rank * rank_num
    data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
    fp_act_cache = ActivationCache(onnx_graph, args, data_st_idx, data_ed_idx)

    input_names = [inp.name for inp in onnx_graph.graph.input]
    output_names = []
    for node in onnx_graph.graph.node:
        for out in node.output:
            output_names.append(out)
    tensor_names = input_names + output_names

    st = time.time()
    for name in tensor_names:
        if name == '':
            continue
        for i in range(data_ed_idx - data_st_idx):
            data_max = fp_act_cache[name][i].max()
            data_min = fp_act_cache[name][i].min()
            # If dynamic_sym = True, Means one more bit.
            if np.abs(data_min - 0) < 1e-6 and 'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']:
                unsigned = 4
            else:
                unsigned = 1
            abs_x = np.abs(fp_act_cache[name][i])
            s_n = abs_x.sum() / abs_x[abs_x > 0].size
            for _ in range(20):
                s_n_plus_1 = abs_x[abs_x > s_n].sum() / \
                    (1 / (4 ** 8) / 3 / unsigned * abs_x[abs_x <= s_n].size + abs_x[abs_x > s_n].size)
                if np.abs(s_n_plus_1 - s_n) < 1e-6:
                    break
                s_n = s_n_plus_1
            if name in statistics:
                statistics[name]['optimal_s'].append(s_n)
                statistics[name]['min'].append(data_min)
                statistics[name]['max'].append(data_max)
            else:
                statistics[name] = {
                    'optimal_s': [s_n],
                    'min': [data_min],
                    'max': [data_max]
                }
    ed = time.time()

    logger.info("Forward time: {:.2f} seconds".format(ed - st))
    return statistics


def input_data_generator(input_dir, input_name_list, data_st_idx, data_ed_idx):
    for idx in range(data_st_idx, data_ed_idx):
        data = {}
        for i in input_name_list:
            data[i] = np.fromfile(f'{input_dir}/{i}/{idx}.bin', 'float32')
        yield data


def forward_get_tensor(graph, net, index, args):
    for node in graph.graph.node:
        if node.op_type in QUANT_NODE_NAME_LIST:
            continue
        for output_name in node.output:
            if output_name not in [_o.name for _o in net.graph.output]:
                net.graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    providers = [("CUDAExecutionProvider", {'device_id': device})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    ort_inputs = {}
    for data in input_data_generator(args.input_dir, graph.network_inputs, index, index + 1):
        for name in graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(graph.get_tensor_shape(name))
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
    return copy.deepcopy(ort_outs)



def forward_get_output(graph, net, index, args):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    providers = [("CUDAExecutionProvider", {'device_id': device})]
    ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    ort_inputs = {}
    for data in input_data_generator(args.batch_data_dir, graph.network_inputs, index, index + 1):
        for name in graph.network_inputs:
            ort_inputs[name] = data[name][:].reshape(graph.get_tensor_shape(name))
        outputs = [output.name for output in ort_session.get_outputs()]
        ort_outputs = ort_session.run(outputs, ort_inputs)
        ort_outs = OrderedDict(zip(outputs, ort_outputs))
    return copy.deepcopy(ort_outs)