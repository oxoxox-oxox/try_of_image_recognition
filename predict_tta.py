import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import os


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
    """
    普通预测函数（无随机性）
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式

    # 图像预处理（预测阶段使用确定性转换）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.CenterCrop(32),  # 使用中心裁剪代替随机裁剪
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
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


def predict_image_with_tta(image_path, model_path, num_classes=10, num_augmentations=10):
    """
    使用测试时数据增强（TTA）进行预测
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 基础转换（无随机性）
    base_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # 增强转换（包含随机性）
    aug_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # 加载图像
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    all_probabilities = []

    with torch.no_grad():
        # 首先使用基础转换（无增强）进行一次预测
        base_tensor = base_transform(image).unsqueeze(0).to(device)
        base_output = model(base_tensor)
        base_probs = F.softmax(base_output, dim=1)
        all_probabilities.append(base_probs)

        # 然后进行多次增强预测
        for i in range(num_augmentations - 1):
            aug_tensor = aug_transform(image).unsqueeze(0).to(device)
            aug_output = model(aug_tensor)
            aug_probs = F.softmax(aug_output, dim=1)
            all_probabilities.append(aug_probs)

    # 合并所有预测结果（平均概率）
    avg_probabilities = torch.mean(torch.cat(all_probabilities, dim=0), dim=0)

    # 获取最终预测结果
    confidence, predicted = torch.max(avg_probabilities, 0)

    return predicted.item(), confidence.item()


def predict_image_ensemble(image_path, model_path, num_classes=10, num_augmentations=5):
    """
    另一种TTA实现：投票机制 + 平均置信度
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 定义多个不同的增强策略
    augmentations = [
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.0),  # 无翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=1.0),  # 强制翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]),
        transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    ]

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    predictions = []
    confidences = []

    with torch.no_grad():
        for transform in augmentations:
            for i in range(num_augmentations):
                tensor = transform(image).unsqueeze(0).to(device)
                output = model(tensor)
                probs = F.softmax(output, dim=1)

                confidence, predicted = torch.max(probs, 1)
                predictions.append(predicted.item())
                confidences.append(confidence.item())

    # 统计每个类别的总置信度
    class_scores = {i: 0 for i in range(num_classes)}
    for pred, conf in zip(predictions, confidences):
        class_scores[pred] += conf

    # 选择总置信度最高的类别
    final_prediction = max(class_scores, key=class_scores.get)
    final_confidence = class_scores[final_prediction] / len(predictions)

    return final_prediction, final_confidence


def predict_folder(folder_path, model_path, num_classes=10, mode="normal"):
    """
    预测文件夹中的所有图像

    参数:
        folder_path: 图像文件夹路径
        model_path: 模型文件路径
        num_classes: 类别数量
        mode: 预测模式 ("normal", "tta", "ensemble")
    """
    # CIFAR-10 的类别名称
    if num_classes == 10:
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    else:
        class_names = [f'类别{i}' for i in range(num_classes)]

    # 统计结果
    total_images = 0
    results = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 仅处理图像文件
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            total_images += 1
            print(f"正在处理图像 ({total_images}): {filename}")

            # 根据模式选择预测方法
            if mode == "tta":
                predicted_class, confidence = predict_image_with_tta(
                    file_path, model_path, num_classes, num_augmentations=8
                )
                method = "TTA增强预测"
            elif mode == "ensemble":
                predicted_class, confidence = predict_image_ensemble(
                    file_path, model_path, num_classes, num_augmentations=5
                )
                method = "集成预测"
            else:
                predicted_class, confidence = predict_image(
                    file_path, model_path, num_classes)
                method = "普通预测"

            result = {
                'filename': filename,
                'predicted_class': predicted_class,
                'class_name': class_names[predicted_class],
                'confidence': confidence,
                'method': method
            }
            results.append(result)

            print(f"图像: {filename}")
            print(f"预测方法: {method}")
            print(f"预测结果: {class_names[predicted_class]}")
            print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
            print("-" * 50)

    # 打印统计信息
    print("\n" + "="*60)
    print(f"预测完成！总共处理了 {total_images} 张图像")
    print(f"预测模式: {mode}")

    # 显示置信度统计
    if results:
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        max_confidence = max(r['confidence'] for r in results)
        min_confidence = min(r['confidence'] for r in results)

        print(f"平均置信度: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print(f"最高置信度: {max_confidence:.4f} ({max_confidence*100:.2f}%)")
        print(f"最低置信度: {min_confidence:.4f} ({min_confidence*100:.2f}%)")

        # 显示类别分布
        print("\n类别分布:")
        class_distribution = {}
        for r in results:
            class_name = r['class_name']
            class_distribution[class_name] = class_distribution.get(
                class_name, 0) + 1

        for class_name, count in class_distribution.items():
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {count} 张 ({percentage:.1f}%)")


def main():
    """
    主函数 - 在这里配置参数并运行预测
    """
    # 指定路径
    folder_path = r'.\data\photo'
    model_path = r'model.pth'
    num_classes = 10

    # 选择预测模式:
    # "normal" - 普通预测（快速，确定性）
    # "tta" - 测试时数据增强（更可靠，较慢）
    # "ensemble" - 集成预测（最可靠，最慢）
    prediction_mode = "tta"

    print(f"开始预测，模式: {prediction_mode}")
    print(f"图像文件夹: {folder_path}")
    print(f"模型文件: {model_path}")
    print("="*60)

    # 调用函数进行遍历识别
    predict_folder(folder_path, model_path, num_classes, mode=prediction_mode)


if __name__ == '__main__':
    main()
