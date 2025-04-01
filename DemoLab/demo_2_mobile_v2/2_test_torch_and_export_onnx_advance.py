import torch, torchvision
import torch.nn as nn
import sys
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import matplotlib.pyplot as plt
import time, os, copy, numpy as np
from tqdm import tqdm
import dipoorlet_utils.quant_config as config
from dipoorlet_utils.dataset import get_dataset
from spectrautils.print_utils import *
import argparse

current_file_path = os.path.dirname(os.path.abspath(__file__))

# model = torch.load(f"{current_file_path}/models/2024_10_30_mobilev2_model.pth")
# 加载checkpoint
checkpoint = torch.load(f"{config.export_work_dir}/mobile_v2_epoch_38_checkpoint.pth")

# 创建模型实例 - 与训练代码保持一致
model = torchvision.models.mobilenet_v2(pretrained=False)

# 修改分类器以适应200类输出
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 200)
)

# 处理DataParallel前缀问题
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
    new_state_dict[name] = v

# 加载处理后的状态字典
model.load_state_dict(new_state_dict)

# 设置为评估模式
model.eval()

# 移动到GPU
model = model.cuda()

_, val_dataset, _ = get_dataset()


dataloaders = torch.utils.data.DataLoader(
    val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=8
)

running_corrects = 0.0
for i, (inputs, labels) in tqdm(enumerate(dataloaders)):
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    running_corrects += torch.sum(preds == labels.data)
print_colored_box(f"Accuracy : {running_corrects / len(val_dataset) * 100:.2f}%")

# convert to onnx
if isinstance(model, torch.nn.DataParallel):
    model = model.module


x = torch.randn(1, 3, 224, 224).cuda()

export_onnx_path = f"{config.export_work_dir}/mobilev2_model_new.onnx"
torch.onnx.export(
    model, 
    x, 
    export_onnx_path, 
    export_params=True, 
    opset_version=11
)
print(f"onnx has been saved in {export_onnx_path}")

def get_args():
    parser = argparse.ArgumentParser(description='测试MobileNetV2模型并导出ONNX')
    parser.add_argument('--checkpoint', type=str, 
                        default=None,
                        help='指定要加载的检查点文件路径')
    parser.add_argument('--batch_size', type=int, default=config.val_batch_size, 
                        help='测试批量大小')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否可视化一些预测结果')
    parser.add_argument('--export_onnx', action='store_true', 
                        help='是否导出ONNX模型')
    parser.add_argument('--onnx_path', type=str, 
                        default=f"{config.export_work_dir}/mobilev2_model.onnx",
                        help='ONNX模型保存路径')
    
    args = parser.parse_args()
    return args

def find_latest_checkpoint(directory):
    """查找目录中最新的检查点文件"""
    if not os.path.exists(directory):
        return None
    
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('_checkpoint.pth')]
    if not checkpoint_files:
        return None
    
    # 按照文件修改时间排序，获取最新的文件
    latest_file = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)

def load_model(checkpoint_path=None):
    """加载模型和检查点"""
    # 如果未指定检查点，查找最新的检查点
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(config.export_work_dir)
        if checkpoint_path is None:
            print_colored_box("未找到检查点文件，请指定有效的检查点路径", text_color='red')
            exit(1)
    
    print_colored_box(f"加载检查点: {checkpoint_path}", text_color='blue')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)
    
    # 创建模型
    model = torchvision.models.mobilenet_v2(pretrained=False)
    
    # 修改分类器以适应200类输出（与训练代码保持一致）
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 200),
    )
    
    # 加载模型状态字典
    if 'model_state_dict' in checkpoint:
        # 处理可能的DataParallel前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 移除'module.'前缀
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        
        # 打印模型信息
        if 'epoch' in checkpoint and 'best_acc' in checkpoint:
            print_colored_box([
                f"模型信息:",
                f"  训练轮次: {checkpoint['epoch']}",
                f"  最佳准确率: {checkpoint['best_acc']:.4f}"
            ], text_color='green')
    else:
        # 如果直接保存的是state_dict
        model.load_state_dict(checkpoint)
    
    return model

def evaluate_model(model, batch_size=32, visualize=False):
    """评估模型性能"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 获取数据集
    _, val_dataset, class_names = get_dataset()
    
    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # 用于存储分类结果
    all_preds = []
    all_labels = []
    class_correct = list(0. for i in range(200))
    class_total = list(0. for i in range(200))
    
    running_corrects = 0.0
    total_samples = 0
    
    # 用于可视化的样本
    vis_images = []
    vis_preds = []
    vis_labels = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 统计总体准确率
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            
            # 统计每个类别的准确率
            correct = (preds == labels).squeeze()
            for j in range(len(labels)):
                label = labels[j]
                class_correct[label] += correct[j].item()
                class_total[label] += 1
            
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 收集可视化样本
            if visualize and i == 0:  # 只收集第一个批次的样本
                vis_images.extend(inputs.cpu())
                vis_preds.extend(preds.cpu().numpy())
                vis_labels.extend(labels.cpu().numpy())
    
    # 计算总体准确率
    overall_accuracy = running_corrects.double() / total_samples
    
    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(200):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            class_accuracies.append((i, class_acc))
    
    # 按准确率排序，找出表现最好和最差的类别
    class_accuracies.sort(key=lambda x: x[1])
    worst_classes = class_accuracies[:5]  # 最差的5个类别
    best_classes = class_accuracies[-5:]  # 最好的5个类别
    
    # 计算混淆矩阵中的错误最多的类别对
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    np.fill_diagonal(cm, 0)  # 忽略对角线（正确分类）
    
    # 找出错误最多的5个类别对
    worst_pairs = []
    for _ in range(5):
        max_idx = np.argmax(cm)
        i, j = max_idx // 200, max_idx % 200
        worst_pairs.append((i, j, cm[i, j]))
        cm[i, j] = 0
    
    # 打印评估结果
    eval_time = time.time() - start_time
    
    print_colored_box([
        f"模型评估结果:",
        f"  总体准确率: {overall_accuracy:.4f}",
        f"  评估样本数: {total_samples}",
        f"  评估时间: {eval_time:.2f}秒"
    ], text_color='green', box_color='yellow')
    
    # print([
    #     f"表现最好的类别:",
    #     *[f"  {class_names[i] if class_names else f'类别 {i}'}: {acc:.4f}" for i, acc in reversed(best_classes)]
    # ])
    
    # print_colored_box([
    #     f"表现最差的类别:",
    #     *[f"  {class_names[i] if class_names else f'类别 {i}'}: {acc:.4f}" for i, acc in worst_classes]
    # ], text_color='red')
    
    # print_colored_box([
    #     f"最容易混淆的类别对:",
    #     *[f"  {class_names[i] if class_names else f'类别 {i}'} → {class_names[j] if class_names else f'类别 {j}'}: {count}次" 
    #       for i, j, count in worst_pairs]
    # ], text_color='yellow')
    
    # 可视化一些预测结果
    if visualize and vis_images:
        visualize_predictions(vis_images, vis_preds, vis_labels, class_names)
    
    return overall_accuracy

def visualize_predictions(images, preds, labels, class_names=None):
    """可视化预测结果"""
    plt.figure(figsize=(15, 10))
    
    # 最多显示16张图片
    num_images = min(16, len(images))
    
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        
        # 转换图像格式
        img = images[i].permute(1, 2, 0).numpy()
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        
        # 获取类别名称
        pred_name = class_names[preds[i]] if class_names else f"类别 {preds[i]}"
        true_name = class_names[labels[i]] if class_names else f"类别 {labels[i]}"
        
        # 设置标题颜色
        color = 'green' if preds[i] == labels[i] else 'red'
        plt.title(f"预测: {pred_name}\n真实: {true_name}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    vis_dir = os.path.join(config.export_work_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(os.path.join(vis_dir, 'predictions.png'))
    print_colored_box(f"可视化结果已保存至: {os.path.join(vis_dir, 'predictions.png')}", text_color='blue')
    
    plt.show()

def export_onnx(model, onnx_path):
    """导出ONNX模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 如果模型是DataParallel，获取内部模型
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 导出ONNX模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'output': {0: 'batch_size'}}
    )
    
    print_colored_box(f"ONNX模型已导出至: {onnx_path}", text_color='blue', box_color='yellow')
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print_colored_box("ONNX模型验证通过!", text_color='green')
    except ImportError:
        print_colored_box("未安装onnx包，跳过模型验证", text_color='yellow')
    except Exception as e:
        print_colored_box(f"ONNX模型验证失败: {str(e)}", text_color='red')

def main():
    args = get_args()
    
    # 加载模型
    model = load_model(args.checkpoint)
    
    # 评估模型
    accuracy = evaluate_model(model, batch_size=args.batch_size, visualize=args.visualize)
    
    # 导出ONNX模型
    if args.export_onnx:
        export_onnx(model, args.onnx_path)

if __name__ == "__main__":
    main()
