import os
import sys
import datetime
import torch
import torch.nn as nn
from spectrautils import print_utils 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time, os, copy, numpy as np
from spectrautils import logging_utils, print_utils
# from dipoorlet_utils import quant_config
from dipoorlet_utils.dataset import get_dataset
import argparse
from torch.utils.data import RandomSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

# 添加命令行参数解析
def get_config():
    parser = argparse.ArgumentParser(description='MobileNetV2训练或继续微调')
    parser.add_argument('--resume', 
                        type=str, 
                        default="",
                        help='加载检查点文件路径继续训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数，如果不指定则使用配置文件中的值')
    parser.add_argument('--lr', type=float, default=0.01, help='初始学习率设置')
    parser.add_argument('--auto_resume', default=False, help='自动加载指定目录中最新的模型文件')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='权重衰减')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑参数')
    parser.add_argument('--mixup', type=float, default=0.2, help='Mixup alpha参数')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='学习率调度器')
    
    args = parser.parse_args()
    return args


logger_manager = logging_utils.AsyncLoggerManager("./logs")
logger = logger_manager.logger

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.SYSTEM.CUDA_IDS
num_gpus = torch.cuda.device_count()


# 添加一个函数来查找最新的模型文件
def find_latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    
    checkpoint_files = [f for f in os.listdir(directory) if f.endswith('_checkpoint.pth')]
    if not checkpoint_files:
        return None
    
    # 按照文件修改时间排序，获取最新的文件
    latest_file = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(directory, x)))
    return os.path.join(directory, latest_file)


# 实现标签平滑的交叉熵损失
class LabelSmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


# 实现Mixup数据增强
def mixup_data(x, y, alpha=0.2, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# 添加EMA模型平均
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 注册模型参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# 改进的训练函数
def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, start_epoch, 
    num_epochs=25, use_mixup=True, mixup_alpha=0.2, use_ema=True
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    
    # 初始化EMA
    if use_ema:
        ema = EMA(model, decay=0.998)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 在每个epoch结束时记录验证准确率，用于提前停止
    val_accs = []
    patience = 5  # 提前停止的耐心值
    patience_counter = 0

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        epoch_start = time.time()
        
        # 每个epoch包含训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式
                if use_ema:
                    ema.apply_shadow()  # 在验证阶段应用EMA权重

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                # 只在训练阶段跟踪历史
                with torch.set_grad_enabled(phase == "train"):
                    if phase == "train" and use_mixup:
                        # 应用Mixup数据增强
                        inputs_mixed, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
                        outputs = model(inputs_mixed)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 仅在训练阶段反向传播+优化
                    if phase == "train":
                        loss.backward()
                        # 梯度裁剪，防止梯度爆炸
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        # 更新EMA
                        if use_ema:
                            ema.update()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                current_mean_loss = running_loss / ((i + 1) * inputs.size(0))
                
                # 在非mixup情况下或验证阶段计算准确率
                if not (phase == "train" and use_mixup):
                    running_corrects += torch.sum(preds == labels.data)
                    current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                else:
                    # 在mixup训练中，我们无法直接计算准确率
                    # 但为了显示进度，我们可以使用非mixup方式计算一个近似值
                    with torch.no_grad():
                        _, preds_direct = torch.max(model(inputs), 1)
                        batch_correct = torch.sum(preds_direct == labels.data)
                        running_corrects += batch_correct
                        current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                
                if i % 50 == 0:
                    logger.info(
                        "{} Epoch:[{}|{}], Iteration: {}/{}, Loss: {:.4f}, Acc: {:.4f}, LR: {:.6f}".format(
                            phase.capitalize(),
                            epoch + 1,
                            num_epochs, 
                            i + 1, 
                            len(dataloaders[phase]), 
                            loss.item(),
                            current_acc,
                            optimizer.param_groups[0]['lr']
                        )
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == "train":
                avg_loss = epoch_loss
                t_acc = epoch_acc
                # 根据调度器类型更新学习率
                if isinstance(scheduler, ReduceLROnPlateau):
                    # 这种调度器需要在验证阶段之后更新
                    pass
                else:
                    scheduler.step()
            else:  # 验证阶段
                val_loss = epoch_loss
                val_acc = epoch_acc
                
                # 在验证阶段后更新ReduceLROnPlateau调度器
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                
                # 恢复EMA
                if use_ema:
                    ema.restore()
                
                # 记录验证准确率用于提前停止
                val_accs.append(val_acc)
                
                # 检查是否需要保存最佳模型
                if epoch_acc > best_acc:
                    logger.info("新的最佳验证准确率! {:.4f} > {:.4f}".format(epoch_acc, best_acc))
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # 确保目录存在
                    os.makedirs(cfg.DIPOORLET.export_work_dir, exist_ok=True)
                    
                    # 保存检查点
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                        'ema_shadow': ema.shadow if use_ema else None,
                    }, f"{cfg.DIPOORLET.export_work_dir}/mobile_v2_epoch_{epoch + 1}_checkpoint.pth")
                    
                    # 重置提前停止计数器
                    patience_counter = 0
                else:
                    patience_counter += 1
                    logger.info(f"验证准确率未提高。提前停止计数: {patience_counter}/{patience}")
                
                # 检查是否需要提前停止
                if patience_counter >= patience:
                    logger.info(f"连续{patience}个epoch验证准确率未提高，提前停止训练。")
                    # 提前结束训练循环
                    epoch_time = time.time() - epoch_start
                    print_utils.print_colored_box([
                        "提前停止训练!",
                        f"  最佳验证准确率: {best_acc:.4f}",
                        f"  在第{epoch + 1 - patience}个epoch获得"
                    ], text_color='red', box_color='yellow')
                    
                    # 加载最佳模型权重
                    model.load_state_dict(best_model_wts)
                    return model

        # 打印每个epoch的摘要
        epoch_time = time.time() - epoch_start
        print()
        epoch_summary = [
            "Epoch Summary:",
            f"  Train Loss: {avg_loss:.4f} | Train Accuracy: {t_acc:.4f}",
            f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}",
            f"  Best Val Accuracy So Far: {best_acc:.4f}",
            f"  Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}",
            f"  Epoch Time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s"
        ]
        
        print_utils.print_colored_box(epoch_summary, text_color='green', box_color='yellow')

    time_elapsed = time.time() - since
    print_utils.print_colored_box(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )
    
    print_utils.print_colored_box(f"Best val Acc: {best_acc:4f}", attrs=['bold'], text_color='green', box_color='yellow')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


def main():

    args = get_config()
    
    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # 修改分类器以适应200类输出
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 200)
    
    # 添加批归一化以提高稳定性
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 200),
    )
    
    # 将模型移至GPU
    model = model.to(device)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    
    # 获取数据集
    train_dataset, val_dataset, _ = get_dataset(cfg.DIPOORLET.imagenet_200_dir)
    
    train_loaders = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.DIPOORLET.train_batch_size, shuffle=True, num_workers=8
    )
    val_loaders = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.DIPOORLET.val_batch_size, shuffle=True, num_workers=8
    )
    # 定义损失函数
    criterion = LabelSmoothCrossEntropyLoss(smoothing=args.label_smoothing)
    
    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    if args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs if args.epochs else cfg.DIPOORLET.epochs)
    else:  # plateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # 设置起始epoch
    start_epoch = 0
    
    # 如果需要从检查点恢复训练
    if args.resume or args.auto_resume:
        checkpoint_path = args.resume
        
        # 如果启用自动恢复，查找最新的检查点
        if args.auto_resume and not isinstance(args.auto_resume, str):
            latest_checkpoint = find_latest_checkpoint(cfg.DIPOORLET.export_work_dir)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                logger.info(f"自动加载最新检查点: {checkpoint_path}")
            else:
                logger.info("未找到检查点，将从头开始训练")
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"从检查点加载: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # 如果检查点包含优化器状态，加载它
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 如果检查点包含调度器状态，加载它
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 设置起始epoch
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                
            logger.info(f"成功加载检查点，从epoch {start_epoch} 继续训练")
        else:
            logger.info("未找到有效的检查点，将从头开始训练")
    
    # 确保导出目录存在
    os.makedirs(cfg.DIPOORLET.export_work_dir, exist_ok=True)
    
    # 训练模型
    epochs = args.epochs if args.epochs else cfg.DIPOORLET.epochs
    logger.info(f"开始训练，总共 {epochs} 个epochs，从epoch {start_epoch} 开始")
    
    # 记录训练配置
    config_summary = [
        "训练配置:",
        f"  学习率: {args.lr}",
        f"  权重衰减: {args.weight_decay}",
        f"  标签平滑: {args.label_smoothing}",
        f"  Mixup: {args.mixup}",
        f"  调度器: {args.scheduler}",
        f"  总epochs: {epochs}",
        f"  起始epoch: {start_epoch}",
        f"  批量大小: {cfg.DIPOORLET.train_batch_size}",
        f"  设备: {device} ({num_gpus} GPUs)" if num_gpus > 1 else f"  设备: {device}"
    ]
    print_utils.print_colored_box(config_summary, text_color='blue', box_color='yellow')
    
    print()
    
    dataloaders = {}
    dataloaders["train"] = train_loaders
    dataloaders["val"] = val_loaders

    dataset_sizes = {}
    dataset_sizes["train"] = len(train_dataset)
    dataset_sizes["val"] = len(val_dataset)


    # 训练模型
    model = train_model(
        model, 
        dataloaders,
        dataset_sizes, 
        criterion, 
        optimizer, 
        scheduler, 
        start_epoch,
        num_epochs=epochs,
        use_mixup=args.mixup > 0,
        mixup_alpha=args.mixup,
        use_ema=True
    )
    
    # 保存最终模型
    final_model_path = os.path.join(cfg.DIPOORLET.export_work_dir, "mobile_v2_final.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型已保存至: {final_model_path}")
    
    
    logger.info("训练完成!")


if __name__ == "__main__":
    main()