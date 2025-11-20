# utils/transforms.py
import torchvision.transforms as transforms

# CIFAR-10 数据集的统计信息
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def get_train_transform(use_advanced_aug=True):
    """获取训练数据transform"""
    if use_advanced_aug:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])

def get_val_transform():
    """获取验证数据transform"""
    return transforms.Compose([
        transforms.Resize((32,32)),
        transforms.CenterCrop(32),  # 使用中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

def get_val_transform_tta():
    """tta版验证集"""
    return  transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

def get_vot_argument_tta():
    return [
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



# 默认使用的transform
train_transform = get_train_transform()
val_transform = get_val_transform()
val_transform_tta = get_val_transform_tta()
vot_argument_tta = get_vot_argument_tta()


