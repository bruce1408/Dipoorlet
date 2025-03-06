import copy
import time
import sys, os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

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

# def forward_net_octav(onnx_graph, args):
#     # 生成图和网络的深拷贝，防止对原始图的修改
#     net = copy.deepcopy(onnx_graph.model)
#     graph = net.graph
    
#     # 确保所有节点的输出都在图的输出列表中
#     for node in reversed(graph.node):
#         for output_name in reversed(node.output):
#             if output_name not in [_o.name for _o in graph.output]:
#                 graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    
#     # 设置CUDA执行提供者
#     providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
#     ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    
#     # 检查CUDA是否被使用
#     if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
#         logger.warning("CUDA可能未被使用。请检查您的ort/cuda/cudnn版本。")

#     # 初始化统计信息字典和计时器
#     statistics = {}
#     t1 = 0
    
#     # 计算每个进程处理的数据范围
#     # rank_num = args.data_num // args.world_size
#     # data_st_idx = args.rank * rank_num
#     # data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
#     data_st_idx = 0
#     data_ed_idx = args.data_num
    
#     # 获取ORT会话的输出名称
#     outputs = [output.name for output in ort_session.get_outputs()]
    
#     # 遍历数据批次，进行前向传播和统计
#     for data_batch in tqdm(input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx),
#                            desc='OCTAV'):
        
#         # 准备ORT输入
#         ort_inputs = {name: data_batch[name][:].reshape(onnx_graph.get_tensor_shape(name)) 
#                       for name in onnx_graph.network_inputs}
        
#         st = time.time()
#         ort_outputs = ort_session.run(outputs, ort_inputs)
#         ed = time.time()
#         t1 += ed - st

#         # 将输出结果与输入合并
#         ort_outs = OrderedDict(zip(outputs, ort_outputs))
#         ort_inputs.update(ort_outs)

#         # 计算每个张量的统计信息
#         for i, tensor in ort_inputs.items():
#             data_max = np.max(tensor)
#             data_min = np.min(tensor)
            
#             # 判断是否使用动态对称量化 如果dynamic_sym = True，意味着多一位
#             unsigned = 4 if (np.abs(data_min) < 1e-6 and 
#                              'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']) else 1

#             abs_x = np.abs(tensor)
#             non_zero_mask = abs_x > 0
#             s_n = abs_x.sum() / np.count_nonzero(non_zero_mask)

#             # 迭代计算最优的s_n
#             for _ in range(20):
#                 mask = abs_x > s_n
#                 s_n_plus_1 = abs_x[mask].sum() / (1 / (4 ** 8) / 3 / unsigned * np.sum(~mask) + np.sum(mask))
#                 if np.abs(s_n_plus_1 - s_n) < 1e-6:
#                     break
#                 s_n = s_n_plus_1

#             # 更新统计信息
#             if i in statistics:
#                 statistics[i]['optimal_s'].append(s_n)
#                 statistics[i]['min'].append(data_min)
#                 statistics[i]['max'].append(data_max)
#             else:
#                 statistics[i] = {
#                     'optimal_s': [s_n],
#                     'min': [data_min],
#                     'max': [data_max]
#                 }
#     # 记录并输出前向传播的总时间
#     logger.info("前向传播时间: {:.2f} 秒".format(t1))
#     return statistics

# 使用内存映射加载数据，减少内存开销并加速加载
def load_data_mmap(input_dir, input_name_list, start_idx, end_idx, batch_size=1):
    """
    使用内存映射批量加载数据，提高 I/O 效率
    
    Args:
        input_dir: 输入目录
        input_name_list: 输入名称列表
        start_idx: 起始索引
        end_idx: 结束索引
        batch_size: 批量大小
    
    Returns:
        生成器，每次生成一批数据
    """
    for batch_idx in range(start_idx, end_idx, batch_size):
        batch_end = min(batch_idx + batch_size, end_idx)
        batch_data = []
        
        for idx in range(batch_idx, batch_end):
            data = {}
            for i in input_name_list:
                file_path = f'{input_dir}/{i}/{idx}.bin'
                if os.path.exists(file_path):
                    data[i] = np.memmap(file_path, dtype='float32', mode='r')
                else:
                    logger.warning(f"文件不存在: {file_path}")
                    # 创建空数组作为备用
                    data[i] = np.array([], dtype='float32')
            batch_data.append(data)
        
        yield batch_data

# 优化版 OCTAV 计算函数
def optimize_octav(abs_x, unsigned=1, max_iterations=20, tolerance=1e-6):
    """
    使用向量化操作优化 OCTAV 算法计算
    
    Args:
        abs_x: 绝对值张量
        unsigned: 无符号因子
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
    
    Returns:
        计算得到的最优比例因子
    """
    # 处理全零张量的情况
    non_zero_mask = abs_x > 0
    non_zero_count = np.count_nonzero(non_zero_mask)
    if non_zero_count == 0:
        return 0
    
    # 初始 s_n 计算
    # 使用 float64 提高精度
    s_n = np.float64(abs_x.sum()) / non_zero_count
    
    # 预计算常数因子
    constant = 1.0 / (4.0 ** 8) / 3.0 / unsigned
    
    # 提前计算张量总和, 避免重复计算
    abs_x_sum = abs_x.sum()
    total_elements = abs_x.size
    
    for _ in range(max_iterations):
        # 创建掩码并计算关键值
        mask = abs_x > s_n
        mask_sum = np.sum(mask)
        
        # 检查边界情况
        if mask_sum == 0:
            break  # 避免除以零
            
        # 使用向量化操作计算新的 s_n
        masked_sum = np.sum(abs_x[mask])
        s_n_plus_1 = masked_sum / (constant * (total_elements - mask_sum) + mask_sum)
        
        # 检查收敛
        if np.abs(s_n_plus_1 - s_n) < tolerance:
            break
            
        s_n = s_n_plus_1
    
    return s_n

"""
# Numba 加速版本的 OCTAV 计算 (取消注释以启用)
@numba.jit(nopython=True)
def optimize_octav_numba(abs_x, unsigned=1, max_iterations=20, tolerance=1e-6):
    # 处理全零张量
    non_zero_count = 0
    for val in abs_x.flat:
        if val > 0:
            non_zero_count += 1
    
    if non_zero_count == 0:
        return 0.0
    
    # 初始 s_n 计算
    abs_sum = 0.0
    for val in abs_x.flat:
        abs_sum += val
    
    s_n = abs_sum / non_zero_count
    
    # 预计算常数
    constant = 1.0 / (4.0 ** 8) / 3.0 / unsigned
    total_elements = abs_x.size
    
    for _ in range(max_iterations):
        # 计算掩码和和
        mask_sum = 0
        masked_abs_sum = 0.0
        
        for val in abs_x.flat:
            if val > s_n:
                mask_sum += 1
                masked_abs_sum += val
        
        # 检查边界情况
        if mask_sum == 0:
            break
            
        # 计算新的 s_n
        s_n_plus_1 = masked_abs_sum / (constant * (total_elements - mask_sum) + mask_sum)
        
        # 检查收敛
        if abs(s_n_plus_1 - s_n) < tolerance:
            break
            
        s_n = s_n_plus_1
    
    return s_n
"""

# 优化的批量处理函数
def process_tensors_batch(all_tensors, args, detailed_timing=False):
    """
    批量处理张量集合，使用向量化操作提高性能
    
    Args:
        all_tensors: 所有张量的字典
        args: 程序参数
        detailed_timing: 是否记录详细时间
    
    Returns:
        张量统计结果的字典
    """
    results = {}
    tensor_timings = {}
    octav_timings = {}
    
    for tensor_name, tensor in all_tensors.items():
        tensor_start = time.time()
        
        # 计算基本统计信息 (一次性完成)
        data_max = np.max(tensor)
        data_min = np.min(tensor)
        abs_tensor = np.abs(tensor)
        
        # 确定无符号因子
        unsigned = 4 if (np.abs(data_min) < 1e-6 and 
                        'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']) else 1
        
        # 计时 OCTAV 计算
        octav_start = time.time()
        # 使用优化的 OCTAV 计算
        s_n = optimize_octav(abs_tensor, unsigned)
        # 或使用 Numba 加速版本
        # s_n = optimize_octav_numba(abs_tensor, unsigned)
        octav_end = time.time()
        
        # 保存结果
        if tensor_name in results:
            results[tensor_name]['optimal_s'].append(s_n)
            results[tensor_name]['min'].append(data_min)
            results[tensor_name]['max'].append(data_max)
        else:
            results[tensor_name] = {
                'optimal_s': [s_n],
                'min': [data_min],
                'max': [data_max]
            }
        
        tensor_end = time.time()
        
        # 记录时间（如果需要）
        if detailed_timing:
            tensor_time = tensor_end - tensor_start
            octav_time = octav_end - octav_start
            
            if tensor_name in tensor_timings:
                tensor_timings[tensor_name] += tensor_time
                octav_timings[tensor_name] += octav_time
            else:
                tensor_timings[tensor_name] = tensor_time
                octav_timings[tensor_name] = octav_time
    
    return results, tensor_timings, octav_timings

# 缓存管理函数
class TensorCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, tensor_name, tensor_hash):
        """获取缓存中的张量统计信息"""
        key = (tensor_name, tensor_hash)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, tensor_name, tensor_hash, stats):
        """将张量统计信息放入缓存"""
        key = (tensor_name, tensor_hash)
        if len(self.cache) >= self.max_size:
            # 简单策略：删除第一个元素
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = stats
    
    def get_stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self.cache),
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

# 主优化函数
def forward_net_octav(onnx_graph, args):
    """
    优化版前向网络 OCTAV 计算函数
    
    Args:
        onnx_graph: ONNX 图
        args: 程序参数
    
    Returns:
        张量统计信息字典
    """
    debug_mode = args.debug if hasattr(args, 'debug') else False
    # 初始化时间统计字典
    timing_stats = {
        'model_preparation': 0,
        'session_creation': 0,
        'data_preparation': 0,
        'inference': 0,
        'statistics_computation': 0,
        'total': 0
    }
    detail_timing = {
        'tensor_processing': {},
        'octav_iterations': {}
    }
    
    # 初始化缓存
    tensor_cache = TensorCache(max_size=1000)
    
    total_start = time.time()
    
    # 1. 模型准备阶段 (避免不必要的深拷贝)
    prep_start = time.time()
    # 仅当需要修改图时才进行深拷贝
    net = copy.deepcopy(onnx_graph.model)
    graph = net.graph
    
    # 确保所有节点的输出都在图的输出列表中
    output_names = set(_o.name for _o in graph.output)
    for node in reversed(graph.node):
        for output_name in reversed(node.output):
            if output_name not in output_names:
                graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
                output_names.add(output_name)
    prep_end = time.time()
    timing_stats['model_preparation'] = prep_end - prep_start
    
    # 2. 会话创建阶段
    session_start = time.time()
    providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
    
    # 序列化一次，避免重复序列化
    serialized_model = net.SerializeToString()
    ort_session = ort.InferenceSession(serialized_model, providers=providers)
    
    # 检查CUDA是否被使用
    if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
        logger.warning("CUDA可能未被使用。请检查您的ort/cuda/cudnn版本。")
    session_end = time.time()
    timing_stats['session_creation'] = session_end - session_start

    # 3. 数据准备阶段
    data_prep_start = time.time()
    # 初始化统计信息字典
    statistics = {}
    
    # 计算处理的数据范围
    data_st_idx = 0
    data_ed_idx = args.data_num
    
    # 预分配内存给统计信息 (提前创建所有必要的字典)
    outputs = [output.name for output in ort_session.get_outputs()]
    all_tensor_names = set(onnx_graph.network_inputs + outputs)
    for tensor_name in all_tensor_names:
        statistics[tensor_name] = {
            'optimal_s': [],
            'min': [],
            'max': []
        }
        detail_timing['tensor_processing'][tensor_name] = 0
        detail_timing['octav_iterations'][tensor_name] = 0
    
    # 使用更大批量的数据加载
    batch_size = min(32, args.data_num)  # 根据内存大小调整批量大小
    data_prep_end = time.time()
    timing_stats['data_preparation'] = data_prep_end - data_prep_start
    
    # 4. 处理阶段
    inference_time = 0
    stats_computation_time = 0
    batch_times = []
    data_loading_times = []
    
    # 使用优化的数据加载器
    for batch_idx, batch_tensors in enumerate(
        load_data_mmap(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx, batch_size)):
        
        batch_time_start = time.time()
        
        # 记录数据加载时间
        data_loading_end = time.time()
        data_loading_times.append(data_loading_end - batch_time_start)
        
        # 批量处理每个数据
        for data_idx, data_batch in enumerate(tqdm(batch_tensors, 
                                             desc=f'OCTAV Batch {batch_idx+1}/{(data_ed_idx-data_st_idx+batch_size-1)//batch_size}')):
            # 准备ORT输入
            ort_inputs = {}
            for name in onnx_graph.network_inputs:
                if name in data_batch and len(data_batch[name]) > 0:
                    ort_inputs[name] = data_batch[name][:].reshape(onnx_graph.get_tensor_shape(name))
            
            # 计算推理时间
            inference_start = time.time()
            ort_outputs = ort_session.run(outputs, ort_inputs)
            inference_end = time.time()
            current_inference_time = inference_end - inference_start
            inference_time += current_inference_time

            # 将输出结果与输入合并
            ort_outs = OrderedDict(zip(outputs, ort_outputs))
            all_tensors = {**ort_inputs, **ort_outs}

            # 统计计算开始
            stats_start = time.time()
            
            # 使用优化的张量处理方法
            batch_stats, tensor_times, octav_times = process_tensors_batch(
                all_tensors, args, detailed_timing=debug_mode)
            
            # 更新全局统计信息
            for tensor_name, tensor_stats in batch_stats.items():
                for key, values in tensor_stats.items():
                    statistics[tensor_name][key].extend(values)
            
            # 如果启用了详细计时，更新时间统计
            if debug_mode:
                for tensor_name, time_value in tensor_times.items():
                    detail_timing['tensor_processing'][tensor_name] += time_value
                for tensor_name, time_value in octav_times.items():
                    detail_timing['octav_iterations'][tensor_name] += time_value
            
            stats_end = time.time()
            stats_computation_time += stats_end - stats_start
        
        batch_time_end = time.time()
        batch_times.append(batch_time_end - batch_time_start)
        
        # 定期显示进度和性能
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            avg_batch_time = sum(batch_times) / len(batch_times)
            est_remaining = avg_batch_time * ((data_ed_idx - data_st_idx + batch_size - 1) // batch_size - (batch_idx + 1))
            logger.info(f"已完成 {batch_idx+1} 批处理，平均每批 {avg_batch_time:.2f} 秒，估计剩余时间 {est_remaining:.2f} 秒")
            
            if tensor_cache.hits + tensor_cache.misses > 0:
                logger.info(f"缓存统计: 命中率 {tensor_cache.hits/(tensor_cache.hits+tensor_cache.misses)*100:.2f}%")
    
    # 5. 更新时间统计
    timing_stats['inference'] = inference_time
    timing_stats['statistics_computation'] = stats_computation_time
    
    total_end = time.time()
    timing_stats['total'] = total_end - total_start
    
    # 6. 生成时间统计报告
    logger.info("========== 时间统计报告 ==========")
    logger.info(f"总运行时间: {timing_stats['total']:.4f} 秒")
    logger.info(f"模型准备时间: {timing_stats['model_preparation']:.4f} 秒 ({timing_stats['model_preparation']/timing_stats['total']*100:.2f}%)")
    logger.info(f"会话创建时间: {timing_stats['session_creation']:.4f} 秒 ({timing_stats['session_creation']/timing_stats['total']*100:.2f}%)")
    logger.info(f"数据准备时间: {timing_stats['data_preparation']:.4f} 秒 ({timing_stats['data_preparation']/timing_stats['total']*100:.2f}%)")
    logger.info(f"推理执行时间: {timing_stats['inference']:.4f} 秒 ({timing_stats['inference']/timing_stats['total']*100:.2f}%)")
    logger.info(f"统计计算时间: {timing_stats['statistics_computation']:.4f} 秒 ({timing_stats['statistics_computation']/timing_stats['total']*100:.2f}%)")
    
    logger.info("\n===== 批处理统计 =====")
    if batch_times:
        logger.info(f"平均批处理时间: {sum(batch_times)/len(batch_times):.4f} 秒")
        logger.info(f"最长批处理时间: {max(batch_times):.4f} 秒")
        logger.info(f"最短批处理时间: {min(batch_times):.4f} 秒")
    if data_loading_times:
        logger.info(f"平均数据加载时间: {sum(data_loading_times)/len(data_loading_times):.4f} 秒")
    
    # 仅在调试模式下显示详细的张量时间信息
    if debug_mode:
        logger.info("\n===== 张量处理时间排名(前10) =====")
        sorted_tensor_times = sorted(detail_timing['tensor_processing'].items(), key=lambda x: x[1], reverse=True)
        for i, (tensor_name, tensor_time) in enumerate(sorted_tensor_times[:10], 1):
            percentage = tensor_time / timing_stats['total'] * 100
            logger.info(f"{i}. {tensor_name}: {tensor_time:.4f} 秒 ({percentage:.2f}%)")
        
        logger.info("\n===== OCTAV迭代时间排名(前10) =====")
        sorted_octav_times = sorted(detail_timing['octav_iterations'].items(), key=lambda x: x[1], reverse=True)
        for i, (tensor_name, octav_time) in enumerate(sorted_octav_times[:10], 1):
            percentage = octav_time / timing_stats['total'] * 100
            logger.info(f"{i}. {tensor_name}: {octav_time:.4f} 秒 ({percentage:.2f}%)")
        
        # 缓存统计
        if tensor_cache.hits + tensor_cache.misses > 0:
            cache_stats = tensor_cache.get_stats()
            logger.info("\n===== 缓存性能 =====")
            logger.info(f"缓存命中: {cache_stats['hits']}")
            logger.info(f"缓存未命中: {cache_stats['misses']}")
            logger.info(f"缓存命中率: {cache_stats['hit_ratio']*100:.2f}%")
            logger.info(f"缓存大小: {cache_stats['size']}/{tensor_cache.max_size}")
    
    return statistics



   

   
# def forward_net_octav(onnx_graph, args):
#     # 初始化时间统计字典
#     timing_stats = {
#         'model_preparation': 0,
#         'session_creation': 0,
#         'data_preparation': 0,
#         'inference': 0,
#         'statistics_computation': 0,
#         'total': 0
#     }
#     detail_timing = {
#         'tensor_processing': {},
#         'octav_iterations': {}
#     }
    
#     total_start = time.time()
    
#     # 生成图和网络的深拷贝，防止对原始图的修改
#     prep_start = time.time()
#     net = copy.deepcopy(onnx_graph.model)
#     graph = net.graph
    
#     # 确保所有节点的输出都在图的输出列表中
#     for node in reversed(graph.node):
#         for output_name in reversed(node.output):
#             if output_name not in [_o.name for _o in graph.output]:
#                 graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
#     prep_end = time.time()
#     timing_stats['model_preparation'] = prep_end - prep_start
    
#     # 设置CUDA执行提供者
#     session_start = time.time()
#     providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
#     ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    
#     # 检查CUDA是否被使用
#     if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
#         logger.warning("CUDA可能未被使用。请检查您的ort/cuda/cudnn版本。")
#     session_end = time.time()
#     timing_stats['session_creation'] = session_end - session_start

#     # 初始化统计信息字典
#     statistics = {}
    
#     # 计算处理的数据范围，不再使用分布式计算
#     data_st_idx = 0
#     data_ed_idx = args.data_num
    
#     # 获取ORT会话的输出名称
#     outputs = [output.name for output in ort_session.get_outputs()]
    
#     # 预分配内存给统计信息
#     data_prep_start = time.time()
#     all_tensor_names = set(onnx_graph.network_inputs + outputs)
#     for tensor_name in all_tensor_names:
#         statistics[tensor_name] = {
#             'optimal_s': [],
#             'min': [],
#             'max': []
#         }
#         detail_timing['tensor_processing'][tensor_name] = 0
#         detail_timing['octav_iterations'][tensor_name] = 0
    
#     # 使用批处理方式加载数据以减少I/O开销
#     batch_size = 16  # 可根据内存大小调整批量大小
    
#     # 创建数据生成器的迭代器
#     data_iterator = input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx)
#     data_prep_end = time.time()
#     timing_stats['data_preparation'] = data_prep_end - data_prep_start
    
#     # 批量处理数据
#     inference_time = 0
#     stats_computation_time = 0
#     batch_times = []
#     data_loading_times = []
    
#     for batch_start in range(data_st_idx, data_ed_idx, batch_size):
#         batch_time_start = time.time()
#         batch_end = min(batch_start + batch_size, data_ed_idx)
#         batch_tensors = []
        
#         # 收集批量数据
#         data_loading_start = time.time()
#         for _ in range(batch_end - batch_start):
#             try:
#                 batch_tensors.append(next(data_iterator))
#             except StopIteration:
#                 break
#         data_loading_end = time.time()
#         data_loading_times.append(data_loading_end - data_loading_start)
        
#         # 批量处理每个数据
#         for data_batch in tqdm(batch_tensors, desc='OCTAV Batch'):
#             # 准备ORT输入
#             ort_inputs = {name: data_batch[name][:].reshape(onnx_graph.get_tensor_shape(name)) 
#                           for name in onnx_graph.network_inputs}
            
#             # 计算推理时间
#             inference_start = time.time()
#             ort_outputs = ort_session.run(outputs, ort_inputs)
#             inference_end = time.time()
#             current_inference_time = inference_end - inference_start
#             inference_time += current_inference_time

#             # 将输出结果与输入合并
#             ort_outs = OrderedDict(zip(outputs, ort_outputs))
#             all_tensors = {**ort_inputs, **ort_outs}

#             # 统计计算开始
#             stats_start = time.time()
            
#             # 使用NumPy的向量化操作批量计算统计信息
#             for tensor_name, tensor in all_tensors.items():
#                 tensor_start = time.time()
                
#                 data_max = np.max(tensor)
#                 data_min = np.min(tensor)
                
#                 # 判断是否使用动态对称量化 如果dynamic_sym = True，意味着多一位
#                 unsigned = 4 if (np.abs(data_min) < 1e-6 and 
#                                 'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']) else 1

#                 # 优化OCTAV计算
#                 abs_x = np.abs(tensor)
#                 non_zero_mask = abs_x > 0
#                 if np.count_nonzero(non_zero_mask) == 0:
#                     s_n = 0  # 处理全零张量的情况
#                 else:
#                     # 初始化s_n
#                     s_n = abs_x.sum() / np.count_nonzero(non_zero_mask)
                    
#                     # 计时OCTAV迭代
#                     octav_start = time.time()
#                     # 使用NumPy的向量化操作进行迭代计算
#                     iterations = 0
#                     for _ in range(20):
#                         iterations += 1
#                         mask = abs_x > s_n
#                         mask_sum = np.sum(mask)
#                         if mask_sum == 0:
#                             break  # 防止除以零
                        
#                         s_n_plus_1 = abs_x[mask].sum() / (1 / (4 ** 8) / 3 / unsigned * np.sum(~mask) + mask_sum)
#                         if np.abs(s_n_plus_1 - s_n) < 1e-6:
#                             break
#                         s_n = s_n_plus_1
#                     octav_end = time.time()
#                     detail_timing['octav_iterations'][tensor_name] += octav_end - octav_start
                
#                 # 更新统计信息
#                 statistics[tensor_name]['optimal_s'].append(s_n)
#                 statistics[tensor_name]['min'].append(data_min)
#                 statistics[tensor_name]['max'].append(data_max)
                
#                 tensor_end = time.time()
#                 detail_timing['tensor_processing'][tensor_name] += tensor_end - tensor_start
            
#             stats_end = time.time()
#             stats_computation_time += stats_end - stats_start
        
#         batch_time_end = time.time()
#         batch_times.append(batch_time_end - batch_time_start)
    
#     # 更新时间统计
#     timing_stats['inference'] = inference_time
#     timing_stats['statistics_computation'] = stats_computation_time
    
#     total_end = time.time()
#     timing_stats['total'] = total_end - total_start
    
#     # 生成更详细的统计报告
#     logger.info("========== 时间统计报告 ==========")
#     logger.info(f"总运行时间: {timing_stats['total']:.4f} 秒")
#     logger.info(f"模型准备时间: {timing_stats['model_preparation']:.4f} 秒 ({timing_stats['model_preparation']/timing_stats['total']*100:.2f}%)")
#     logger.info(f"会话创建时间: {timing_stats['session_creation']:.4f} 秒 ({timing_stats['session_creation']/timing_stats['total']*100:.2f}%)")
#     logger.info(f"数据准备时间: {timing_stats['data_preparation']:.4f} 秒 ({timing_stats['data_preparation']/timing_stats['total']*100:.2f}%)")
#     logger.info(f"推理执行时间: {timing_stats['inference']:.4f} 秒 ({timing_stats['inference']/timing_stats['total']*100:.2f}%)")
#     logger.info(f"统计计算时间: {timing_stats['statistics_computation']:.4f} 秒 ({timing_stats['statistics_computation']/timing_stats['total']*100:.2f}%)")
    
#     logger.info("\n===== 批处理统计 =====")
#     logger.info(f"平均批处理时间: {sum(batch_times)/len(batch_times):.4f} 秒")
#     logger.info(f"最长批处理时间: {max(batch_times):.4f} 秒")
#     logger.info(f"最短批处理时间: {min(batch_times):.4f} 秒")
#     logger.info(f"平均数据加载时间: {sum(data_loading_times)/len(data_loading_times):.4f} 秒")
    
#     logger.info("\n===== 张量处理时间排名(前10) =====")
#     sorted_tensor_times = sorted(detail_timing['tensor_processing'].items(), key=lambda x: x[1], reverse=True)
#     for i, (tensor_name, process_time) in enumerate(sorted_tensor_times[:10]):
#         logger.info(f"{i+1}. {tensor_name}: {process_time:.4f} 秒 ({process_time/timing_stats['statistics_computation']*100:.2f}%)")
    
#     logger.info("\n===== OCTAV迭代时间排名(前10) =====")
#     sorted_octav_times = sorted(detail_timing['octav_iterations'].items(), key=lambda x: x[1], reverse=True)
#     for i, (tensor_name, octav_time) in enumerate(sorted_octav_times[:10]):
#         logger.info(f"{i+1}. {tensor_name}: {octav_time:.4f} 秒 ({octav_time/timing_stats['statistics_computation']*100:.2f}%)")
    
#     return statistics



# def forward_get_minmax_transformer(onnx_graph, args):
#     # Start minmax activation quantization.
#     statistics = {}
#     rank_num = args.data_num // args.world_size
#     data_st_idx = args.rank * rank_num
#     data_ed_idx = min((args.rank + 1) * rank_num, args.data_num)
#     fp_act_cache = ActivationCache(onnx_graph, args, data_st_idx, data_ed_idx)

#     input_names = [inp.name for inp in onnx_graph.graph.input]
#     output_names = []
#     for node in onnx_graph.graph.node:
#         for out in node.output:
#             output_names.append(out)
#     tensor_names = input_names + output_names

#     st = time.time()
#     for name in tensor_names:
#         if name == '':
#             continue
#         for i in range(data_ed_idx - data_st_idx):
#             if name in statistics:
#                 statistics[name]['max'].append(fp_act_cache[name][i].max())
#                 statistics[name]['min'].append(fp_act_cache[name][i].min())
#             else:
#                 statistics[name] = {}
#                 statistics[name]['max'] = [fp_act_cache[name][i].max()]
#                 statistics[name]['min'] = [fp_act_cache[name][i].min()]
#     ed = time.time()

#     logger.info("Forward time: {:.2f} seconds".format(ed - st))
#     return statistics


# def forward_net_octav(onnx_graph, args):
#     # 生成图和网络的深拷贝，防止对原始图的修改
#     net = copy.deepcopy(onnx_graph.model)
#     graph = net.graph
    
#     # 确保所有节点的输出都在图的输出列表中
#     for node in reversed(graph.node):
#         for output_name in reversed(node.output):
#             if output_name not in [_o.name for _o in graph.output]:
#                 graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    
#     # 设置CUDA执行提供者
#     providers = [("CUDAExecutionProvider", {'device_id': args.local_rank})]
#     ort_session = ort.InferenceSession(net.SerializeToString(), providers=providers)
    
#     # 检查CUDA是否被使用
#     if 'CUDAExecutionProvider' not in ort_session.get_provider_options():
#         logger.warning("CUDA可能未被使用。请检查您的ort/cuda/cudnn版本。")

#     # 初始化统计信息字典和计时器
#     statistics = {}
#     t1 = 0
    
#     # 计算每个进程处理的数据范围
#     # rank_num = args.data_num // args.world_size
#     data_st_idx = 0
#     data_ed_idx = args.data_num
    
#     # 获取ORT会话的输出名称
#     outputs = [output.name for output in ort_session.get_outputs()]
    
#     # 预分配内存给统计信息
#     all_tensor_names = set(onnx_graph.network_inputs + outputs)
#     for tensor_name in all_tensor_names:
#         statistics[tensor_name] = {
#             'optimal_s': [],
#             'min': [],
#             'max': []
#         }
    
#     # 使用批处理方式加载数据以减少I/O开销
#     batch_size = 16  # 可根据内存大小调整批量大小
    
#     # 创建数据生成器的迭代器
#     data_iterator = input_data_generator(args.input_dir, onnx_graph.network_inputs, data_st_idx, data_ed_idx)
    
#     # 批量处理数据
#     for batch_start in range(data_st_idx, data_ed_idx, batch_size):
#         batch_end = min(batch_start + batch_size, data_ed_idx)
#         batch_tensors = []
        
#         # 收集批量数据
#         for _ in range(batch_end - batch_start):
#             try:
#                 batch_tensors.append(next(data_iterator))
#             except StopIteration:
#                 break
        
#         # 批量处理每个数据
#         for data_batch in tqdm(batch_tensors, desc='OCTAV Batch'):
#             # 准备ORT输入
#             ort_inputs = {name: data_batch[name][:].reshape(onnx_graph.get_tensor_shape(name)) 
#                           for name in onnx_graph.network_inputs}
            
#             st = time.time()
#             ort_outputs = ort_session.run(outputs, ort_inputs)
#             ed = time.time()
#             t1 += ed - st

#             # 将输出结果与输入合并
#             ort_outs = OrderedDict(zip(outputs, ort_outputs))
#             all_tensors = {**ort_inputs, **ort_outs}

#             # 使用NumPy的向量化操作批量计算统计信息
#             for tensor_name, tensor in all_tensors.items():
#                 data_max = np.max(tensor)
#                 data_min = np.min(tensor)
                
#                 # 判断是否使用动态对称量化 如果dynamic_sym = True，意味着多一位
#                 unsigned = 4 if (np.abs(data_min) < 1e-6 and 
#                                 'dynamic_sym' in platform_setting_table[args.deploy]['qi_params']) else 1

#                 # 优化OCTAV计算
#                 abs_x = np.abs(tensor)
#                 non_zero_mask = abs_x > 0
#                 if np.count_nonzero(non_zero_mask) == 0:
#                     s_n = 0  # 处理全零张量的情况
#                 else:
#                     # 初始化s_n
#                     s_n = abs_x.sum() / np.count_nonzero(non_zero_mask)
                    
#                     # 使用NumPy的向量化操作进行迭代计算
#                     for _ in range(20):
#                         mask = abs_x > s_n
#                         mask_sum = np.sum(mask)
#                         if mask_sum == 0:
#                             break  # 防止除以零
                        
#                         s_n_plus_1 = abs_x[mask].sum() / (1 / (4 ** 8) / 3 / unsigned * np.sum(~mask) + mask_sum)
#                         if np.abs(s_n_plus_1 - s_n) < 1e-6:
#                             break
#                         s_n = s_n_plus_1
                
#                 # 更新统计信息
#                 statistics[tensor_name]['optimal_s'].append(s_n)
#                 statistics[tensor_name]['min'].append(data_min)
#                 statistics[tensor_name]['max'].append(data_max)
    
#     # 记录并输出前向传播的总时间
#     logger.info("前向传播时间: {:.2f} 秒".format(t1))
#     return statistics


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


# def input_data_generator(input_dir, input_name_list, data_st_idx, data_ed_idx):
#     for idx in range(data_st_idx, data_ed_idx):
#         data = {}
#         for i in input_name_list:
#             data[i] = np.fromfile(f'{input_dir}/{i}/{idx}.bin', 'float32')
#         yield data

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os

def read_file(file_path):
    return np.fromfile(file_path, dtype='float32')

def load_data(input_dir, input_name_list, idx):
    data = {}
    for i in input_name_list:
        file_path = f'{input_dir}/{i}/{idx}.bin'
        data[i] = read_file(file_path)
    return data


def input_data_generator(input_dir, input_name_list, data_st_idx, data_ed_idx):
    with ThreadPoolExecutor(max_workers=6) as executor:  # 增加最大线程数
        # 提交所有任务
        futures = [executor.submit(load_data, input_dir, input_name_list, idx) for idx in range(data_st_idx, data_ed_idx)]
        
        # 获取所有结果
        for future in futures:
            yield future.result()



def forward_get_tensor(graph, net, index, args):
    for node in graph.graph.node:
        if node.op_type in QUANT_NODE_NAME_LIST:
            continue
        for output_name in node.output:
            if output_name not in [_o.name for _o in net.graph.output]:
                net.graph.output.insert(0, onnx.ValueInfoProto(name=output_name))
    # rank = dist.get_rank()
    rank = 0
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