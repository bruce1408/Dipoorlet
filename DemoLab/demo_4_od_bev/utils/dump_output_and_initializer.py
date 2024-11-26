import onnx
import json

def get_tensor_info(tensor):
    # 获取张量的 bitwidth 和 dtype
    # ONNX 不直接提供 bitwidth 和 dtype，因此这里使用假设值
    bitwidth = 16  # 假设 bitwidth
    dtype = "float"  # 假设 dtype
    return {"bitwidth": bitwidth, "dtype": dtype}

def dump_model_info(onnx_model_path, output_json_path):
    model = onnx.load(onnx_model_path)
    graph = model.graph

    activation_encodings = {}
    param_encodings = {}

    # 提取所有节点的输出
    for node in graph.node:
        for output in node.output:
            if output not in activation_encodings:
                activation_encodings[output] = [get_tensor_info(output)]
    
    # 提取初始化器 (initializers)
    for initializer in graph.initializer:
        param_name = initializer.name
        if param_name not in param_encodings:
            param_encodings[param_name] = [get_tensor_info(param_name)]

    # 创建 JSON 数据
    model_info = {
        "activation_encodings": activation_encodings,
        "param_encodings": param_encodings
    }

    # 将 JSON 数据写入文件
    with open(output_json_path, 'w') as json_file:
        json.dump(model_info, json_file, indent=4)

# 使用示例
onnx_model_path = '/mnt/share_disk/bruce_trie/onnx_models/modelv5_extract_head.onnx'
output_json_path = "/mnt/share_disk/bruce_trie/onnx_models/modelv5_extract_head.json"
dump_model_info(onnx_model_path, output_json_path)