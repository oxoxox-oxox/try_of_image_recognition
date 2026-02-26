# train_face_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import sys

# 将项目根目录添加到Python搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入你的模型（需要根据人脸识别调整）
from src.models.cnn_model import ImprovedCNN
# 注意：FaceDataset 类需要创建，当前不存在
# from data.custom_dataset import FaceDataset

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    """获取数据增强变换"""
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集的数据增强（只做必要的变换）
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def validate_model(model, val_loader, device):
    """验证模型"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    # 计算每个类别的准确率
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions))
    
    return accuracy

def train_model():
    """训练人脸识别模型"""
    set_seed()
    
    # 训练参数
    epochs = 50  # 增加epoch数，因为数据量少
    batch_size = 8  # 减小batch size
    lr = 0.001
    weight_decay = 1e-4
    patience = 10  # 早停耐心值
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    train_transform, val_transform = get_transforms()
    
    # 创建完整数据集
    # 注意：FaceDataset 类需要创建，当前使用占位符
    print("错误：FaceDataset 类不存在，请创建 data/custom_dataset.py 文件并实现 FaceDataset 类")
    return
    # dataset = FaceDataset(
    #     root_dir='data',
    #     transform=train_transform,
    #     train=True,
    #     train_ratio=0.8
    # )
    
    # 获取类别数量
    num_classes = len(dataset.classes)
    class_names = dataset.get_class_names()
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 获取数据大小
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80%训练
    val_size = dataset_size - train_size  # 20%验证
    
    # 划分训练集和验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 更新验证集的数据变换
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 创建模型
    model = ImprovedCNN(num_classes=num_classes).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 早停机制
    best_accuracy = 0
    patience_counter = 0
    
    # 训练历史记录
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    print("\n开始训练...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            accuracy = 100 * correct / total
            avg_loss = running_loss / (pbar.n + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # 验证
        val_accuracy = validate_model(model, val_loader, device)
        print(f"验证准确率: {val_accuracy:.2f}%")
        
        # 记录历史
        train_loss_history.append(running_loss / len(train_loader))
        train_acc_history.append(100 * correct / total)
        val_acc_history.append(val_accuracy)
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'class_names': class_names,
                'num_classes': num_classes
            }, 'best_face_model.pth')
            print(f"新最佳模型保存！准确率: {best_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= patience:
            print(f"\n早停触发！在 epoch {epoch+1} 停止训练")
            break
        
        # 更新学习率
        scheduler.step()
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_face_model.pth')
    
    # 绘制训练曲线
    plot_training_history(train_loss_history, train_acc_history, val_acc_history)
    
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"模型已保存为: best_face_model.pth")
    print(f"类别信息: {class_names}")

def plot_training_history(train_loss, train_acc, val_acc):
    """绘制训练历史"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(train_loss, label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_acc, label='训练准确率')
        ax2.plot(val_acc, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100)
        plt.show()
        print("训练历史图已保存为 training_history.png")
    except:
        print("无法绘制训练历史图，请安装matplotlib")

# 推理代码
class FaceRecognizer:
    def __init__(self, model_path='best_face_model.pth'):
        """初始化人脸识别器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.class_names = checkpoint['class_names']
        num_classes = checkpoint['num_classes']
        
        # 创建模型
        self.model = ImprovedCNN(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"模型加载成功！类别数: {num_classes}")
        print(f"类别: {self.class_names}")
    
    def predict(self, image):
        """预测单张图片"""
        from PIL import Image
        
        # 预处理
        if isinstance(image, str):  # 如果是文件路径
            image = Image.open(image).convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # 返回结果
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        # 获取所有类别的概率
        all_probs = probabilities.squeeze().cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_value,
            'all_probabilities': dict(zip(self.class_names, all_probs))
        }
    
    def predict_batch(self, image_paths):
        """批量预测"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            results.append((img_path, result))
        return results

if __name__ == '__main__':
    # 检查数据集是否存在
    if not os.path.exists('data'):
        print("请先创建 face_dataset 文件夹，并按照以下结构放置图片：")
        print("face_dataset/")
        print("├── person_01/")
        print("│   ├── 001.jpg")
        print("│   └── ...")
        print("├── person_02/")
        print("│   ├── 001.jpg")
        print("│   └── ...")
        print("└── ...")
        print("\n每个文件夹代表一个人，里面放该人的照片（至少2张）")
    else:
        # 检查是否有足够的类别
        classes = [d for d in os.listdir('data') 
                  if os.path.isdir(os.path.join('data', d))]
        
        if len(classes) < 2:
            print(f"错误：至少需要2个类别，当前只有 {len(classes)} 个")
            print("请在 face_dataset 文件夹下创建至少2个子文件夹")
        else:
            # 检查每个类别的图片数量
            for cls in classes:
                cls_dir = os.path.join('data', cls)
                images = [f for f in os.listdir(cls_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) < 2:
                    print(f"警告：类别 {cls} 只有 {len(images)} 张图片，建议至少2张")
            
            print(f"数据集检查通过，共 {len(classes)} 个类别")
            train_model()