#!/usr/bin/env python3
"""
读取ONNX文件并分析卷积层的权重和偏置
统计每个channel的min/max值，用于量化分析
"""

import onnx
import onnxruntime as ort
import numpy as np
import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional
import json

def load_onnx_model(onnx_path: str):
    """加载ONNX模型"""
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX文件不存在: {onnx_path}")
    
    print(f"加载ONNX模型: {onnx_path}")
    model = onnx.load(onnx_path)
    
    # 验证模型
    try:
        onnx.checker.check_model(model)
        print("✓ ONNX模型验证通过")
    except onnx.checker.ValidationError as e:
        print(f"⚠ ONNX模型验证警告: {e}")
    
    return model

def analyze_conv_layers(model) -> Dict:
    """分析模型中的所有卷积层"""
    conv_layers = {}
    
    # 获取所有权重初始化器
    initializers = {init.name: init for init in model.graph.initializer}
    
    # 遍历所有节点，查找卷积节点
    for node in model.graph.node:
        if node.op_type == 'Conv':
            layer_name = node.name if node.name else f"Conv_{len(conv_layers)}"
            print(f"\n发现卷积层: {layer_name}")
            
            # 获取输入名称
            weight_name = node.input[1] if len(node.input) > 1 else None
            bias_name = node.input[2] if len(node.input) > 2 else None
            
            weight_data = None
            bias_data = None
            
            # 提取权重
            if weight_name and weight_name in initializers:
                weight_init = initializers[weight_name]
                weight_data = onnx.numpy_helper.to_array(weight_init)
                print(f"  权重: {weight_name}, 形状: {weight_data.shape}")
            else:
                print(f"  警告: 未找到权重 {weight_name}")
            
            # 提取偏置
            if bias_name and bias_name in initializers:
                bias_init = initializers[bias_name]
                bias_data = onnx.numpy_helper.to_array(bias_init)
                print(f"  偏置: {bias_name}, 形状: {bias_data.shape}")
            else:
                print(f"  信息: 未找到偏置 {bias_name}")
            
            conv_layers[layer_name] = {
                'node': node,
                'weight_name': weight_name,
                'bias_name': bias_name,
                'weight_data': weight_data,
                'bias_data': bias_data,
                'weight_shape': weight_data.shape if weight_data is not None else None,
                'bias_shape': bias_data.shape if bias_data is not None else None
            }
    
    return conv_layers

def analyze_weight_distribution(weight_data: np.ndarray, layer_name: str) -> Dict:
    """分析权重数据的分布"""
    if weight_data is None:
        return {}
    
    print(f"\n分析权重分布 - {layer_name}:")
    
    # 整体统计
    stats = {
        'global_min': float(np.min(weight_data)),
        'global_max': float(np.max(weight_data)),
        'global_mean': float(np.mean(weight_data)),
        'global_std': float(np.std(weight_data)),
        'global_abs_max': float(np.max(np.abs(weight_data))),
        'num_elements': int(np.prod(weight_data.shape))
    }
    
    print(f"  全局最小值: {stats['global_min']:.6f}")
    print(f"  全局最大值: {stats['global_max']:.6f}")
    print(f"  全局绝对值最大值: {stats['global_abs_max']:.6f}")
    print(f"  均值: {stats['global_mean']:.6f}, 标准差: {stats['global_std']:.6f}")
    print(f"  元素总数: {stats['num_elements']}")
    
    # 按输出channel分析 (对于卷积权重，形状通常是 [out_channels, in_channels, height, width])
    if len(weight_data.shape) == 4:  # 标准2D卷积
        out_channels = weight_data.shape[0]
        in_channels = weight_data.shape[1]
        
        print(f"  形状: {weight_data.shape} (输出通道={out_channels}, 输入通道={in_channels})")
        
        # 每个输出channel的统计
        channel_stats = []
        for oc in range(out_channels):
            channel_weights = weight_data[oc, :, :, :].flatten()
            channel_min = float(np.min(channel_weights))
            channel_max = float(np.max(channel_weights))
            channel_abs_max = float(np.max(np.abs(channel_weights)))
            
            channel_stats.append({
                'channel': oc,
                'min': channel_min,
                'max': channel_max,
                'abs_max': channel_abs_max,
                'range': channel_max - channel_min
            })
        
        stats['per_output_channel'] = channel_stats
        
        # 输出前5个channel的统计
        print(f"  前5个输出通道的统计:")
        for i in range(min(5, out_channels)):
            cs = channel_stats[i]
            print(f"    通道 {i}: min={cs['min']:.6f}, max={cs['max']:.6f}, abs_max={cs['abs_max']:.6f}")
    
    # 计算直方图（用于可视化）
    hist, bin_edges = np.histogram(weight_data.flatten(), bins=50)
    stats['histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    return stats

def analyze_bias_distribution(bias_data: np.ndarray, layer_name: str) -> Dict:
    """分析偏置数据的分布"""
    if bias_data is None:
        return {}
    
    print(f"\n分析偏置分布 - {layer_name}:")
    
    stats = {
        'global_min': float(np.min(bias_data)),
        'global_max': float(np.max(bias_data)),
        'global_mean': float(np.mean(bias_data)),
        'global_std': float(np.std(bias_data)),
        'global_abs_max': float(np.max(np.abs(bias_data))),
        'num_elements': int(np.prod(bias_data.shape))
    }
    
    print(f"  全局最小值: {stats['global_min']:.6f}")
    print(f"  全局最大值: {stats['global_max']:.6f}")
    print(f"  全局绝对值最大值: {stats['global_abs_max']:.6f}")
    print(f"  均值: {stats['global_mean']:.6f}, 标准差: {stats['global_std']:.6f}")
    print(f"  元素总数: {stats['num_elements']}")
    
    # 偏置通常是1D数组，每个输出通道一个值
    if len(bias_data.shape) == 1:
        print(f"  形状: {bias_data.shape} (输出通道数={bias_data.shape[0]})")
        
        # 每个通道的偏置值
        per_channel = []
        for i, bias_val in enumerate(bias_data):
            per_channel.append({
                'channel': i,
                'value': float(bias_val),
                'abs_value': float(abs(bias_val))
            })
        
        stats['per_channel'] = per_channel
        
        # 输出前10个通道的偏置值
        print(f"  前10个通道的偏置值:")
        for i in range(min(10, len(bias_data))):
            print(f"    通道 {i}: {bias_data[i]:.6f}")
    
    return stats

def save_results_to_json(results: Dict, output_path: str):
    """将分析结果保存为JSON文件"""
    # 将numpy数组转换为Python原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n分析结果已保存到: {output_path}")

def generate_summary_report(conv_layers: Dict, output_dir: str):
    """生成摘要报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("卷积层权重和偏置分析报告")
    report_lines.append("=" * 80)
    
    for layer_name, layer_info in conv_layers.items():
        report_lines.append(f"\n卷积层: {layer_name}")
        
        if layer_info['weight_data'] is not None:
            weight_stats = analyze_weight_distribution(layer_info['weight_data'], layer_name)
            report_lines.append(f"  权重形状: {layer_info['weight_shape']}")
            report_lines.append(f"  权重范围: [{weight_stats.get('global_min', 0):.6f}, {weight_stats.get('global_max', 0):.6f}]")
            report_lines.append(f"  权重绝对值最大值: {weight_stats.get('global_abs_max', 0):.6f}")
        
        if layer_info['bias_data'] is not None:
            bias_stats = analyze_bias_distribution(layer_info['bias_data'], layer_name)
            report_lines.append(f"  偏置形状: {layer_info['bias_shape']}")
            report_lines.append(f"  偏置范围: [{bias_stats.get('global_min', 0):.6f}, {bias_stats.get('global_max', 0):.6f}]")
            report_lines.append(f"  偏置绝对值最大值: {bias_stats.get('global_abs_max', 0):.6f}")
    
    report_text = "\n".join(report_lines)
    
    # 保存报告到文件
    report_path = os.path.join(output_dir, "conv_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n摘要报告已保存到: {report_path}")
    
    # 同时在控制台输出
    print(report_text)

def main():
    parser = argparse.ArgumentParser(description='分析ONNX模型中的卷积层权重和偏置')
    parser.add_argument('--onnx',
                        type=str, 
                        default="/home/bruce_ultra/workspace/Quantization_Optimization/Quantizer-Tools/_outputs/models/resnet18.onnx", 
                        help='ONNX模型文件路径')
    
    parser.add_argument('--output', 
                        type=str, 
                        default='./conv_analysis', help='输出目录路径')
    
    parser.add_argument('--save-json', default=True, help='保存详细结果到JSON文件')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # 1. 加载ONNX模型
        model = load_onnx_model(args.onnx)
        
        # 2. 分析卷积层
        conv_layers = analyze_conv_layers(model)
        
        if not conv_layers:
            print("未找到卷积层！")
            return
        
        print(f"\n共发现 {len(conv_layers)} 个卷积层")
        
        # 3. 详细分析每个卷积层
        all_results = {}
        for layer_name, layer_info in conv_layers.items():
            print(f"\n{'='*60}")
            print(f"详细分析: {layer_name}")
            print(f"{'='*60}")
            
            layer_results = {}
            
            # 分析权重
            if layer_info['weight_data'] is not None:
                weight_stats = analyze_weight_distribution(layer_info['weight_data'], layer_name)
                layer_results['weight'] = weight_stats
            
            # 分析偏置
            if layer_info['bias_data'] is not None:
                bias_stats = analyze_bias_distribution(layer_info['bias_data'], layer_name)
                layer_results['bias'] = bias_stats
            
            all_results[layer_name] = layer_results
        
        # 4. 生成摘要报告
        generate_summary_report(conv_layers, args.output)
        
        # 5. 保存详细结果到JSON（如果指定）
        if args.save_json:
            json_path = os.path.join(args.output, "conv_analysis_details.json")
            save_results_to_json(all_results, json_path)
        
        print(f"\n{'='*60}")
        print("分析完成！")
        print(f"输出目录: {args.output}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()