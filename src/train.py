import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import argparse
from tqdm import tqdm

from models.cnn_model import ImprovedCNN
from utils.transforms import train_transform   # 使用独立的 transform


def train_model():

    # 1. 读取超参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step'], default='cosine')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3. 模型、损失、优化器
    model = ImprovedCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. 学习率调度器
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.scheduler == 'cosine'
        else optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    )

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        scheduler.step()

    # 6. 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print("\n训练完成！模型已保存为 model.pth")


if __name__ == '__main__':
    train_model()
