import copy
import json
import logging
import os
import time
import sys

import numpy as np
import onnx
import torch.distributed as dist
from onnx import TensorProto, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.quant_utils import QuantizationMode, QuantType
from termcolor import colored


try:
    # 尝试直接导入，适用于当前目录运行
    from platform_settings import platform_setting_table
except ImportError:
    # 如果直接导入失败，尝试使用相对导入，适用于跨目录调用
    from .platform_settings import platform_setting_table



logger = logging.getLogger("dipoorlet")


class ONNXGraph(object):
    """ONNX模型图处理类
    用于加载、操作和保存ONNX模型图，提供了模型图的各类操作接口

    Attributes:
        model: ONNX模型对象
        graph: ONNX模型图对象
        output_dir: 输出目录
        deploy: 部署平台配置
        model_type: 模型类型
        initializer: 初始化器字典 (name -> (initializer, index))
        input_map: 输入映射字典 (input_name -> list of nodes)
        output_map: 输出映射字典 (output_name -> node)
        network_inputs: 网络输入列表
        network_outputs: 网络输出列表
        tensor_name_shape_map: 张量名称到形状的映射
        value_name_type_map: 值名称到类型的映射
        name_idx_map: 节点名称到索引的映射
        input: 所有输入列表
        output: 所有输出列表
    """
    def __init__(self, model=None, output_dir="", deploy=None, model_type=None):
        """初始化ONNXGraph对象
        
        Args:
            model (optional): ONNX模型对象，默认为None
            output_dir (str): 输出目录路径，默认为空字符串
            deploy (optional): 部署平台配置，默认为None
            model_type (optional): 模型类型，默认为None
        """
        self.model = model
        self.output_dir = output_dir
        self.deploy = deploy
        self.model_type = model_type
        
        # 非输入和输出节点的参数
        self.initializer = {}
        
        
        self.input_map = {}
        
        
        self.output_map = {}
        
        # 模型输入
        self.network_inputs = []
        
        # 模型输出
        self.network_outputs = []
        
        # 模型输入和输出的shape字典
        self.tensor_name_shape_map = {}
        
        # 模型输入、输出、节点输入和输出的类型
        self.value_name_type_map = {}
        
        # 非模型输入、输出的节点名称索引
        self.name_idx_map = {}
        
        # 模型输入以及各个节点输入
        self.input = []
        
        # 模型输出以及各个节点的输出
        self.output = []
        
        if self.model:
            self.graph = self.model.graph
            self._initialize_model()
            
    
    def _initialize_model(self):
        """执行模型初始化操作"""
        # 把node没有名字的节点设置为 类型_idx 的名字
        self.set_names()
        
        # 把constant节点变成initialier节点
        self.convert_constant_to_init()

        # 构建graph的拓扑结构
        self.topologize_graph()
        
        # initializer 设置名字、initializer的键值对字典
        self.prepare_initializer()
        
        # 设置node名称索引的映射
        self.set_index()
        
        # 获取graph 输入、输出、中间节点输入和输出
        self.get_inp_oup()
        
        # 获取模型中所有张量的形状和类型信息，并将这些信息存储在类的属性中
        self.get_shape_type()

    def set_names(self):
        """
        为ONNX模型图中没有名称的节点设置默认名称。
        遍历所有节点,如果节点没有名称,则根据其操作类型和索引生成一个唯一的名称。
        这有助于提高模型的可读性和可调试性。
        """
        for idx, node in enumerate(self.model.graph.node):
            if not node.name:
                node.name = f"{node.op_type}_{idx}"

    def convert_constant_to_init(self):
        """
        将Constant节点转换为初始化器
        """
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                # node.attribute[0].t 是 Constant 类型节点的第一个属性，这个属性存储了一个 TensorProto 对象，表示该常量节点的张量数据。
                tensor = onnx.numpy_helper.to_array(node.attribute[0].t)
                self.set_initializer(node.output[0], tensor, raw=True)

    def prepare_initializer(self):
        """
        准备初始化器字典
        """
        self.initializer.clear()
        for idx, init in enumerate(self.graph.initializer):
            self.initializer[init.name] = (init, idx)

    def get_inp_oup(self):
        """
        整理 ONNX 模型的输入输出信息，包括网络输入、网络输出、中间输入和输出
        """
        self.network_inputs.clear()
        self.network_outputs.clear()
        self.tensor_name_shape_map.clear()
        self.input.clear()
        self.output.clear()
        
        # 处理网络输入
        for input in self.graph.input:
            if isinstance(self.get_tensor_producer(input.name), str) and \
                    input.name not in self.initializer:
                self.network_inputs.append(input.name)
        
        # 处理网络输出
        for output in self.graph.output:
            self.network_outputs.append(output.name)
        self.input = self.network_inputs.copy()
        self.output = self.network_outputs.copy()

        # 处理中间节点的输入输出
        for node in self.model.graph.node:
            for inp in node.input:
                if inp in self.initializer and inp not in self.input:
                    self.input.append(inp)
            for oup in node.output:
                if oup not in self.output:
                    self.output.append(oup)

    def get_shape_type(self):
        """
        获取张量的形状和类型信息
        """
        
        # 处理输入
        for input in self.graph.input:
            if input.name in self.network_inputs:
                # 获取模型输入的形状和类型信息
                self.tensor_name_shape_map[input.name] = [x.dim_value for x in input.type.tensor_type.shape.dim]
                self.value_name_type_map[input.name] = input.type.tensor_type.elem_type

        # 处理输出
        for output in self.graph.output:
            # 获取模型输出的形状和类型信息
            self.tensor_name_shape_map[output.name] = [x.dim_value for x in output.type.tensor_type.shape.dim]
            self.value_name_type_map[output.name] = output.type.tensor_type.elem_type

        
        # 获取当前的模型权重参数，把TensorProto转换成numpy
        for init in self.initializer:
            self.tensor_name_shape_map[init] = list(self.get_initializer(init).shape)
        
        # 处理中间结果
        inferred_value_info = self.model.graph.value_info
        for info in inferred_value_info:
            shape = [x.dim_value for x in info.type.tensor_type.shape.dim]
            value_type = info.type.tensor_type.elem_type
            self.tensor_name_shape_map[info.name] = shape
            self.value_name_type_map[info.name] = value_type

        # 处理量化相关的张量
        value_names = list(self.tensor_name_shape_map.keys())
        for name in value_names:
            self.tensor_name_shape_map[name + "_q"] = self.tensor_name_shape_map[name]
            if self.deploy is not None:
                if name in self.initializer:
                    symmetric = platform_setting_table[self.deploy]['qw_params']['symmetric']
                else:
                    symmetric = platform_setting_table[self.deploy]['qi_params']['symmetric']
                self.value_name_type_map[name + "_q"] = TensorProto.INT8 if symmetric else TensorProto.UINT8
                self.tensor_name_shape_map[name + "_dq"] = self.tensor_name_shape_map[name]
                self.value_name_type_map[name + "_dq"] = TensorProto.FLOAT

    
    def get_tensor_shape(self, tensor_name):
        return self.tensor_name_shape_map[tensor_name]

    def get_value_type(self, value_name):
        return self.value_name_type_map[value_name]

    def get_constant(self, name):
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] == name:
                    return numpy_helper.to_array(node.attribute[0].t).tolist()

    def get_initializer(self, initializer_name):
        # 把TensorProto 转换成numpy 数据格式
        return numpy_helper.to_array(self.initializer[initializer_name][0])

    def set_initializer(self, initializer_name, value_tensor, raw=True):
        idx = None
        data_type = None
        if initializer_name in self.initializer:
            idx = self.initializer[initializer_name][1]
        if raw:
            initializer = numpy_helper.from_array(value_tensor)
        else:
            if value_tensor.dtype == np.float32:
                data_type = TensorProto.FLOAT
                
            if value_tensor.dtype == np.uint8:
                data_type = TensorProto.UINT8
            
            if value_tensor.dtype == np.int8:
                data_type = TensorProto.INT8
            
            initializer = onnx.helper.make_tensor(name=initializer_name,
                                                  data_type=data_type,
                                                  dims=[] if value_tensor.size == 1 else list(value_tensor.shape),
                                                  vals=value_tensor,
                                                  raw=False)
        initializer.name = initializer_name
        if idx is not None:
            self.graph.initializer.remove(self.graph.initializer[idx])
            self.graph.initializer.insert(idx, initializer)
        else:
            self.graph.initializer.append(initializer)
        self.prepare_initializer()

    def topologize_graph(self):
        
        # 清空现有的输入和输出映射
        self.input_map.clear()
        self.output_map.clear()
        
        # 遍历图中的所有节点
        for idx, node in enumerate(self.graph.node):
            # 处理节点的输出
            for output_name in node.output:
                # 将输出张量名称映射到当前节点
                self.output_map[output_name] = node
            # 处理节点的输入
            for input_name in node.input:
                # 如果输入张量名称不在映射中，初始化一个空列表
                if input_name not in self.input_map:
                    self.input_map[input_name] = []
                # 将当前节点添加到消费该输入的节点列表中
                self.input_map[input_name].append(node)

    def get_tensor_producer(self, output_name):
        if output_name not in self.output_map:
            return 'INPUT_TOKEN'
        return self.output_map[output_name]

    def get_tensor_consumer(self, input_name):
        if input_name not in self.input_map:
            return ['OUTPUT_TOKEN']
        return self.input_map[input_name]

    def save_onnx_model(self, name='tmp', size_threshold=2048):
        if self.model_type is not None:
            convert_model_to_external_data(self.model, all_tensors_to_one_file=True,
                                           location="{}.data".format(name),
                                           size_threshold=size_threshold,
                                           convert_attribute=False)

        model_path = os.path.join(self.output_dir, '{}.onnx'.format(name))
        onnx.save(self.model, model_path)

    def remove_node_purely(self, node):
        self.graph.node.remove(node)

    def insert_node_purely(self, node, idx=0):
        self.graph.node.insert(idx, node)

    def insert_qnodes_purely(self, q_nodes, idx=0, node=None):
        node_list = reversed(q_nodes.node)
        if node:
            idx = self.index(node)
        for node in node_list:
            self.graph.node.insert(idx, node)
        for init in q_nodes.initializer:
            self.graph.initializer.append(init)
        self.set_index()

    def del_network_output(self, out_name):
        idx = self.network_outputs.index(out_name)
        self.graph.output.pop(idx)
        self.network_outputs.remove(out_name)

    def add_network_output(self, out_put):
        self.graph.output.append(out_put)
        self.network_outputs.append(out_put.name)

    def del_initializer(self, initializer_name):
        if initializer_name in self.initializer:
            del self.initializer[initializer_name]

    def set_index(self):
        for node_idx, node in enumerate(self.graph.node):
            self.name_idx_map[node.name] = node_idx

    def index(self, node):
        return self.name_idx_map[node.name]

    def update_model(self):
        self.set_index()
        self.model = onnx.helper.make_model(self.graph,
                                            producer_name='updated_model',
                                            opset_imports=self.model.opset_import)
        self.prepare_initializer()
    
    
    def update_model_dim(self, dim=32):
        """
        更新模型的维度信息
        """
        del self.graph.value_info[:]
        for input in self.graph.input:
            if input.name in self.network_inputs:
                dim1 = input.type.tensor_type.shape.dim[0]
                dim1.dim_value = dim

        for output in self.graph.output:
            if output.name in self.network_outputs:
                dim1 = output.type.tensor_type.shape.dim[0]
                dim1.dim_value = dim

        self.update_model()
        self.model = onnx.shape_inference.infer_shapes(self.model)
        self.get_shape_type()
        

    def copy_from(self, source_graph):
        """_summary_
        深度拷贝源图的属性和数据到当前对象。
        该方法用于将一个 ONNXGraph 对象的所有属性和数据复制到另一个对象中，
        确保两个对象完全独立，修改不会相互影响。
        
        Args:
            source_graph (_type_): _description_
        """
        # 复制模型结构
        self.model = copy.deepcopy(source_graph.model)  # 复制整个 ONNX 模型
        self.graph = copy.deepcopy(source_graph.graph)  # 复制计算图
        self.initializer = copy.deepcopy(source_graph.initializer)  # 复制初始化器（常量，如权重）

        # 复制输入输出映射
        self.input_map = copy.deepcopy(source_graph.input_map)  # 输入张量到节点的映射
        self.output_map = copy.deepcopy(source_graph.output_map)  # 输出张量到节点的映射

        # 复制网络输入输出信息
        self.network_inputs = copy.deepcopy(source_graph.network_inputs)  # 网络输入名称列表
        self.network_outputs = copy.deepcopy(source_graph.network_outputs)  # 网络输出名称列表

        # 复制张量形状和类型信息
        self.tensor_name_shape_map = copy.deepcopy(source_graph.tensor_name_shape_map)  # 张量名称到形状的映射
        self.value_name_type_map = copy.deepcopy(source_graph.value_name_type_map)  # 张量名称到类型的映射

        # 复制输入输出名称列表
        self.input = copy.deepcopy(source_graph.input)  # 所有输入名称列表
        self.output = copy.deepcopy(source_graph.output)  # 所有输出名称列表
            
        # 复制节点索引映射和其他配置
        self.name_idx_map = source_graph.name_idx_map.copy()  # 节点名称到索引的映射
        self.output_dir = source_graph.output_dir  # 输出目录路径
        self.deploy = source_graph.deploy  # 部署平台配置
        self.model_type = source_graph.model_type  # 模型类型


def setup_logger(args):
    """配置并返回日志记录器
    
    Args:
        args: 命令行参数对象

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    global logger
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    logger.setLevel(logging.INFO)
    logger_file = os.path.join(args.output_dir,
                               'log-{}.txt'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())))
    with open(logger_file, 'w') as f:
        f.write(str(args) + '\n')
    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)


def cos_similarity(ta, tb):
    assert ta.shape == tb.shape
    if np.sum(ta * tb) == 0:
        return 0.
    return np.sum(ta * tb) / np.sqrt(np.square(ta).sum()) \
        / np.sqrt(np.square(tb).sum())

def max_abs_gap(ta, tb):
    assert ta.shape == tb.shape
    return np.max(np.abs(ta - tb))

# 定义装饰器函数
def dispatch_functool(func):
    registry = {}

    def dispatch(value):
        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):
        if func is None:
            return lambda f: register(value, f)
        registry[value] = func
        return func

    def wrapper(*args, **kw):
        return dispatch(args[0])(*(args[1:]), **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


def update_model_path(name, args):
    '''Update model path saved in args. Often sync among GPUs.
    Always followed load_graph.
    '''
    args.model = os.path.join(args.output_dir, '{}.onnx'.format(name))


def save_clip_val(act_clip_val, weight_clip_val, args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    for k, v in act_clip_val.items():
        act_clip_val[k][0] = act_clip_val[k][0].tolist()
        act_clip_val[k][1] = act_clip_val[k][1].tolist()
    for k, v in weight_clip_val.items():
        weight_clip_val[k][0] = weight_clip_val[k][0].tolist()
        weight_clip_val[k][1] = weight_clip_val[k][1].tolist()
    with open(os.path.join(args.output_dir, act_fname), 'w') as f:
        json.dump(act_clip_val, f, indent=4)
    with open(os.path.join(args.output_dir, weight_fname), 'w') as f:
        json.dump(weight_clip_val, f, indent=4)


def reduce_clip_val(rank_size, args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    '''Collect activation clip val from each GPU and reduce. Weight range use rank0.
    '''
    act_clip_val, weight_clip_val = load_clip_val(args, act_fname + '.rank0', weight_fname + '.rank0')
    for k, v in act_clip_val.items():
        if args.act_quant != 'minmax':
            v[0] /= float(rank_size)
            v[1] /= float(rank_size)
    for i in range(1, rank_size):
        with open(os.path.join(args.output_dir, act_fname + '.rank{}'.format(i)), 'r') as f:
            _act_clip_val = json.load(f)
            for k, v in _act_clip_val.items():
                if args.act_quant != 'minmax':
                    act_clip_val[k][0] += v[0] / float(rank_size)
                    act_clip_val[k][1] += v[1] / float(rank_size)
                else:
                    act_clip_val[k] = [
                        np.array(min(v[0], act_clip_val[k][0])),
                        np.array(max(v[1], act_clip_val[k][1]))]
    save_clip_val(act_clip_val, weight_clip_val, args)


def load_clip_val(args, act_fname='act_clip_val.json', weight_fname='weight_clip_val.json'):
    act_clip_val = {}
    weight_clip_val = {}
    with open(os.path.join(args.output_dir, act_fname), 'r') as f:
        act_clip_val = json.load(f)
        for k, v in act_clip_val.items():
            # We need scalar here.
            act_clip_val[k][0] = np.float64(act_clip_val[k][0])
            act_clip_val[k][1] = np.float64(act_clip_val[k][1])
    with open(os.path.join(args.output_dir, weight_fname), 'r') as f:
        per_channel = False
        if 'per_channel' in platform_setting_table[args.deploy]['qw_params']:
            per_channel = platform_setting_table[args.deploy]['qw_params']['per_channel']
        weight_clip_val = json.load(f)
        for k, v in weight_clip_val.items():
            weight_clip_val[k][0] = np.array(weight_clip_val[k][0])
            weight_clip_val[k][1] = np.array(weight_clip_val[k][1])
            if not per_channel:
                weight_clip_val[k][0] = np.float64(weight_clip_val[k][0])
                weight_clip_val[k][1] = np.float64(weight_clip_val[k][1])
    return act_clip_val, weight_clip_val


def save_profiling_res(layer_cosine_dict, model_cosine_dict, args,
                       layer_res_fname='layer_res.json', model_res_fname='model_res.json'):
    rank = dist.get_rank()
    for k, v in layer_cosine_dict.items():
        layer_cosine_dict[k] = float(v)
    for k, v in model_cosine_dict.items():
        model_cosine_dict[k][0] = float(v[0])
        model_cosine_dict[k][1] = float(v[1])
    if len(layer_cosine_dict) != 0:
        with open(os.path.join(args.output_dir, layer_res_fname + '.rank{}'.format(rank)), 'w') as f:
            json.dump(layer_cosine_dict, f, indent=4)
    with open(os.path.join(args.output_dir, model_res_fname + '.rank{}'.format(rank)), 'w') as f:
        json.dump(model_cosine_dict, f, indent=4)


def reduce_profiling_res(rank_size, args, layer_res_fname='layer_res.json', model_res_fname='model_res.json'):
    '''Collect profiling res from each GPU and reduce.
    '''
    if args.model_type is None:
        with open(os.path.join(args.output_dir, layer_res_fname + '.rank0'), 'r') as f:
            layer_cosine_dict = json.load(f)
    else:
        layer_cosine_dict = {}
    with open(os.path.join(args.output_dir, model_res_fname + '.rank0'), 'r') as f:
        model_cosine_dict = json.load(f)
    if args.model_type is None:
        for k, v in layer_cosine_dict.items():
            layer_cosine_dict[k] = v / float(rank_size)
        for i in range(1, rank_size):
            with open(os.path.join(args.output_dir, layer_res_fname + '.rank{}'.format(i)), 'r') as f:
                _layer_cosine_dict = json.load(f)
                for k, v in _layer_cosine_dict.items():
                    layer_cosine_dict[k] += v / float(rank_size)
    for k, v in model_cosine_dict.items():
        model_cosine_dict[k][0] = v[0] / float(rank_size)
    for i in range(1, rank_size):
        with open(os.path.join(args.output_dir, model_res_fname + '.rank{}'.format(i)), 'r') as f:
            _model_cosine_dict = json.load(f)
            for k, v in _model_cosine_dict.items():
                model_cosine_dict[k][0] += v[0] / float(rank_size)
                model_cosine_dict[k][1] = min(model_cosine_dict[k][1], v[1])
    return layer_cosine_dict, model_cosine_dict


def deploy_QOperator(model, tensor_range, args):
    mode = QuantizationMode.QLinearOps
    per_channel = platform_setting_table[args.deploy]['qw_params']['per_channel']
    op_types_to_quantize = platform_setting_table[args.deploy]['quant_nodes']

    if platform_setting_table[args.deploy]['qw_params']['symmetric']:
        weight_type = QuantType.QInt8
    else:
        weight_type = QuantType.QUInt8

    if platform_setting_table[args.deploy]['qi_params']['symmetric']:
        activation_type = QuantType.QInt8
    else:
        activation_type = QuantType.QUInt8

    quantizer = ONNXQuantizer(model, per_channel, False, mode, True,
                              weight_type, activation_type, tensor_range,
                              None, args.skip_layers, op_types_to_quantize)
    quantizer.quantize_model()
    model_output = os.path.join(args.output_dir, 'qop_model.onnx')
    quantizer.model.save_model_to_file(model_output)
    
def input_data_generator(input_dir, input_name_list, data_st_idx, data_ed_idx):
    for idx in range(data_st_idx, data_ed_idx):
        data = {}
        for i in input_name_list:
            data[i] = np.fromfile(f'{input_dir}/{i}/{idx}.bin', 'float32')
        yield data



def restore_data(args, input_name_list, batch_size=32):

    if os.path.exists(args.batch_data_dir):
        logger.info("True")
        os.system(f"rm -rf {args.batch_data_dir}")

    os.mkdir(args.batch_data_dir)

    for name in input_name_list:
        os.mkdir(args.batch_data_dir + '/' + name)
        batch_data = []
        for idx in range(0, args.data_num):
            data = np.fromfile(f'{args.input_dir}/{name}/{idx}.bin', 'float32').reshape(1, -1)
            batch_data.append(data)
            if (idx + 1) % batch_size == 0:
                batch_id = int(idx / batch_size)
                batch_data = np.vstack(batch_data)
                batch_data.tofile(f'{args.batch_data_dir}/{name}/{batch_id}.bin')
                batch_data = []
                
                
if __name__ == '__main__':
    model_path = "/mnt/share_disk/bruce_trie/onnx_models/resnet50.onnx"
    output_dir = "/mnt/share_disk/bruce_trie/Quantizer-Tools/outputs/dipoorlet_log/3_dipoorlet_models_od_bev/od_bev_adround"
    deploy = "snpe"
    model_type = None
    model = onnx.load(model_path)
    onnx_grpah = ONNXGraph(model, output_dir, deploy=deploy, model_type=model_type)