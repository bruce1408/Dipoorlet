import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 导入AIMET库
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.utils import QuantParams, create_fake_quant_params
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model

# 定义一个简单的分类网络
class ClassificationNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassificationNet, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 数据加载函数
def get_dataloaders(batch_size=64):
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# 训练函数
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch}, Batch: {batch_idx+1}, Loss: {running_loss/100:.3f}, '
                  f'Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
            
    return 100. * correct / total

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.3f}, Test Acc: {accuracy:.2f}%')
    return accuracy

# 应用AIMET的QAT函数
def apply_qat(model, train_loader, device):
    # 1. 执行BatchNorm折叠（可选，但推荐）
    folded_model, _ = fold_all_batch_norms(model, input_shapes=(1, 3, 32, 32))
    
    # 2. 应用跨层均衡（可选，但推荐）
    equalize_model(folded_model)
    
    # 3. 创建量化参数配置
    quant_params = QuantParams(
        weight_bw=8,            # 权重量化位宽
        act_bw=8,               # 激活量化位宽
        round_mode="nearest",   # 舍入模式
        quant_scheme="tf_enhanced"  # 量化方案
    )
    
    # 4. 创建量化模拟模型（QAT模型）
    sim = QuantizationSimModel(
        model=folded_model,
        quant_scheme=quant_params.quant_scheme,
        default_output_bw=quant_params.act_bw,
        default_param_bw=quant_params.weight_bw,
        default_data_type=torch.int8
    )
    
    # 5. 计算激活范围（校准）
    # 选择一小部分数据用于校准
    num_batches = 10
    sim.compute_encodings(
        forward_pass_callback=lambda model: calibration_callback(model, train_loader, num_batches),
        forward_pass_callback_args=None
    )
    
    return sim

# 校准回调函数
def calibration_callback(model, train_loader, num_batches):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            _ = model(inputs)

# 保存量化模型
def save_quantized_model(sim, path):
    # 导出量化模型（可部署格式）
    sim.export(path=path, filename_prefix="quantized_classification")
    print(f"量化模型已保存到: {path}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 超参数
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01
    num_classes = 10
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders(batch_size)
    
    # 创建模型
    model = ClassificationNet(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # 训练浮点模型（预训练）
    print("开始浮点模型预训练...")
    for epoch in range(5):  # 先训练几个epoch作为预训练
        train(model, train_loader, criterion, optimizer, device, epoch)
        evaluate(model, test_loader, criterion, device)
        scheduler.step()
    
    # 保存浮点模型
    torch.save(model.state_dict(), "float_model.pth")
    print("浮点模型预训练完成并保存")
    
    # 应用AIMET的QAT
    print("开始应用QAT...")
    qat_model = apply_qat(model, train_loader, device)
    
    # QAT训练
    print("开始QAT训练...")
    # QAT模型使用较小的学习率
    qat_optimizer = optim.SGD(qat_model.model.parameters(), lr=learning_rate/10, momentum=0.9, weight_decay=5e-4)
    qat_scheduler = optim.lr_scheduler.CosineAnnealingLR(qat_optimizer, T_max=200)
    
    best_acc = 0
    for epoch in range(num_epochs):
        train(qat_model.model, train_loader, criterion, qat_optimizer, device, epoch)
        accuracy = evaluate(qat_model.model, test_loader, criterion, device)
        qat_scheduler.step()
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            save_quantized_model(qat_model, "./quantized_models")
    
    print("QAT训练完成！")
    print(f"最佳精度: {best_acc:.2f}%")

if __name__ == "__main__":
    main()