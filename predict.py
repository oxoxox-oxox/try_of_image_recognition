import os
import torch
from PIL import Image
from src.models.cnn_model import ImprovedCNN
from src.utils.transforms import test_transform
import torch.nn.functional as F


def predict_image(image_path, model_path, num_classes=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = ImprovedCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. 处理图片
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = test_transform(image).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()

    return predicted, confidence


def predict_folder(folder_path, model_path, num_classes=10):
    # CIFAR-10 类名
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n正在处理：{filename}")

            pred, conf = predict_image(file_path, model_path, num_classes)
            print(f"预测结果：{class_names[pred]}")
            print(f"置信度：{conf:.4f} ({conf*100:.2f}%)")


def main():
    folder_path = r'./data/photo'
    model_path = r'model.pth'
    num_classes = 10

    predict_folder(folder_path, model_path, num_classes)


if __name__ == "__main__":
    main()
