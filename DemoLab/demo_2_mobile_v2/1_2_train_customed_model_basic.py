import os
import sys
import datetime
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
import time, os, copy, numpy as np
from spectrautils import logging_utils, print_utils
# from dipoorlet_utils import quant_config
from dipoorlet_utils.dataset import get_dataset
from common.configs import get_cfg_defaults
from spectrautils import logging_utils, print_utils

cfg = get_cfg_defaults()

# 添加命令行参数解析
def get_config():
    parser = argparse.ArgumentParser(description='MobileNetV2训练或继续微调')
    parser.add_argument('--resume', 
                        type=str, 
                        default="",
                        help='加载检查点文件路径继续训练')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数，如果不指定则使用配置文件中的值')
    parser.add_argument('--lr', type=float, default=0.0005, help='初始学习率设置')  # 降低默认学习率
    parser.add_argument('--auto_resume', default=True, help='自动加载指定目录中最新的模型文件')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='选择优化器')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['step', 'plateau', 'cosine'], help='学习率调度器')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小，如不指定则使用配置文件中的值')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减系数')

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

def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, start_epoch, num_epochs=25
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 添加早停机制
    patience = 10  # 连续10个epoch没有提升就早停
    no_improve_epochs = 0
    
    # 记录训练历史
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(start_epoch, num_epochs):
        print("-" * 10)
        epoch_start = time.time()
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                current_acc = running_corrects.double() / ((i + 1) * inputs.size(0))
                
                if i % 50 == 0:
                    logger.info(
                        "{} Epoch:[{}/{}], Iteration: {}/{}, Loss: {:.4f}, Accuracy: {:.4f}".format(
                            phase.capitalize(),
                            epoch + 1,
                            num_epochs, 
                            i + 1, 
                            len(dataloaders[phase]), 
                            loss.item(),
                            current_acc
                        )
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == "train":
                avg_loss = epoch_loss
                t_acc = epoch_acc
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # 根据验证集准确率调整学习率
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_acc)
                elif not isinstance(scheduler, lr_scheduler.StepLR):  # 如果不是StepLR，在验证后调整
                    scheduler.step()

            if phase == "val" and epoch_acc > best_acc:
                logger.info("验证集准确率在第{}轮后: {:.4f}".format(epoch + 1, epoch_acc))
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                os.makedirs(cfg.DIPOORLET.export_work_dir, exist_ok=True)
                
                # 保存每个epoch的模型权重
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                    'history': history,  # 保存训练历史
                }, f"{cfg.DIPOORLET.export_work_dir}/mobile_v2_epoch_{epoch + 1}_checkpoint.pth")
                
                # 重置早停计数器
                no_improve_epochs = 0
            elif phase == "val":
                # 增加早停计数器
                no_improve_epochs += 1

        # 如果使用StepLR，在每个epoch结束后调整
        if isinstance(scheduler, lr_scheduler.StepLR):
            scheduler.step()
            
        # 用列表存储所有的输出信息
        epoch_time = time.time() - epoch_start
        print()
        epoch_summary = [
            "Epoch Summary:",
            "  Train Loss: {:.4f} | Train Accuracy: {:.4f}".format(avg_loss, t_acc),
            "  Val Loss: {:.4f} | Val Accuracy: {:.4f}".format(val_loss, val_acc),
            "  Best Val Accuracy So Far: {:.4f}".format(best_acc),
            "  Epoch Time: {:.0f}m {:.0f}s".format(epoch_time // 60, epoch_time % 60),
            f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
        ]
        
        print_utils.print_colored_box(epoch_summary, text_color='green', box_color='yellow')
        
        # 早停检查
        if no_improve_epochs >= patience:
            logger.info(f"早停: {patience}个epoch没有提升，停止训练")
            break

    time_elapsed = time.time() - since
    print_colored_box(
        "训练完成，总用时 {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    
    print_utils.print_colored_box("最佳验证准确率: {:4f}".format(best_acc), attrs=['bold'], text_color='green', box_color='yellow')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    args = get_config()
    
    # 加载模型
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # 修改分类器以适应200类输出
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 200)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if num_gpus > 1:
        # 使用所有可见的 GPU
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info(f"使用 {num_gpus} 个GPU进行训练: {device_ids}")
    else:
        logger.info("只有一个GPU或没有GPU可用，使用单GPU或CPU。")
    
    model = model.to(device)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 选择优化器
    if args.optimizer.lower() == 'adam':
        optimizer_ft = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logger.info(f"使用Adam优化器，学习率={args.lr}，权重衰减={args.weight_decay}")
    else:
        optimizer_ft = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        logger.info(f"使用SGD优化器，学习率={args.lr}，动量=0.9，权重衰减={args.weight_decay}")
    
    # 选择学习率调度器
    if args.scheduler.lower() == 'plateau':
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer_ft, mode='max', factor=0.5, patience=3, verbose=True
        )
        logger.info("使用ReduceLROnPlateau学习率调度器")
    elif args.scheduler.lower() == 'cosine':
        exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer_ft, T_max=10, eta_min=1e-6
        )
        logger.info("使用CosineAnnealingLR学习率调度器")
    else:
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        logger.info("使用StepLR学习率调度器，每7个epoch降低学习率")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 获取数据集
    train_dataset, val_dataset, _ = get_dataset(cfg.DIPOORLET.imagenet_200_dir)
    
    # 设置批次大小
    train_batch_size = args.batch_size if args.batch_size else cfg.DIPOORLET.train_batch_size
    val_batch_size = args.batch_size if args.batch_size else cfg.DIPOORLET.val_batch_size
    
    train_loaders = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loaders = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    
    dataloaders = {
        "train": train_loaders,
        "val": val_loaders
    }
    
    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset)
    }
    
    # 如果指定了 resume 参数，加载检查点
    start_epoch = 0
    
    # 自动查找最新的检查点文件
    if args.auto_resume and os.path.isdir(cfg.DIPOORLET.export_work_dir):
        latest_checkpoint = find_latest_checkpoint(cfg.DIPOORLET.export_work_dir)
        if latest_checkpoint:
            args.resume = latest_checkpoint
            logger.info(f"自动加载最新的检查点文件: {latest_checkpoint}")
        else:
            logger.info(f"在目录 {cfg.DIPOORLET.export_work_dir} 中未找到检查点文件，将从头开始训练。")
    
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
        exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        logger.info(f"从epoch {start_epoch} 继续训练，当前最佳准确率: {best_acc:.4f}")
    else:
        logger.info("从头开始训练。")
    
    # 如果命令行传入了 epochs 参数，则覆盖配置文件中的 epoch 数值
    num_epochs = args.epochs if args.epochs is not None else cfg.DIPOORLET.epochs
    
    # 训练模型
    model = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        start_epoch,
        num_epochs=num_epochs,
    )
    
    # 保存最终模型
    current_timestamp = datetime.datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y_%m_%d")
    torch.save(model, f"{cfg.DIPOORLET.export_work_dir}/{formatted_timestamp}_mobilev2_model.pth")
    logger.info(f"最终模型已保存到 {cfg.DIPOORLET.export_work_dir}/{formatted_timestamp}_mobilev2_model.pth")

if __name__ == "__main__":
    main()