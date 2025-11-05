import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import torch.nn as nn
import torch.nn.functional as F
import os


# 假设你的 SimpleCNN 定义在同一个文件中，或者从其他模块导入
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 增加卷积层和通道数
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 修改全连接层
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict_image(image_path, model_path, num_classes=10):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 图像预处理（必须与训练时相同）
    transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
    
    # 加载和预处理图像
    image = Image.open(image_path)
    # 如果图像是灰度图，转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # 获取预测结果
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


def predict_folder(folder_path, model_path, num_classes=10):
    # CIFAR-10 的类别名称
    if num_classes == 10:
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    else:
        class_names = [f'类别{i}' for i in range(num_classes)]

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 仅处理图像文件
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"正在处理图像: {filename}")
            
            # 进行预测
            predicted_class, confidence = predict_image(file_path, model_path, num_classes)
            
            print(f"图像: {filename}")
            print(f"预测结果: {class_names[predicted_class]}")
            print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)\n")


def main():
    #指定路径
    folder_path = r'.\data\photo'
    model_path = r'model.pth'
    num_classes = 10

    #调用函数进行遍历识别
    predict_folder(folder_path,model_path,num_classes)

if __name__ == '__main__':
    main()