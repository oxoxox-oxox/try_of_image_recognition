# 图像识别项目

## 项目简介

基于深度学习的图像分类系统，使用PyTorch实现。

## 项目结构

```
try_of_image_recognition/
├── .github/             # GitHub配置文件
│   └── workflows/       # CI/CD工作流
├── data/                # 数据集目录
│   ├── cifar-10-batches-py/  # CIFAR-10数据集
│   └── photo/           # 测试图像
├── docs/                # 文档目录
│   └── usage.md         # 使用说明
├── scripts/             # 脚本目录
│   └── setup.sh         # 安装脚本
├── src/                 # 源代码目录
│   ├── models/          # 模型定义
│   │   ├── __init__.py
│   │   ├── cnn_model.py     # CNN模型
│   │   └── model_utils.py   # 模型工具函数
│   ├── opencv_utils/    # OpenCV相关工具
│   │   ├── __init__.py
│   │   └── source.py        # 人脸检测脚本
│   ├── predict/         # 预测相关脚本
│   │   ├── __init__.py
│   │   ├── predict.py       # 普通预测
│   │   └── predict_tta.py   # 带TTA的预测
│   ├── utils/           # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py        # 配置文件
│   │   └── transforms.py    # 数据变换
│   ├── __init__.py
│   ├── train.py         # 训练脚本
│   └── train_face_model.py  # 人脸识别模型训练
├── tests/               # 测试目录
│   ├── __init__.py
│   └── test_models.py   # 模型测试
├── .gitignore           # Git忽略文件
├── LICENSE              # 许可证
├── README.md            # 项目说明
├── requirements.txt     # 依赖包
└── training_history.png # 训练历史图
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

### 训练模型

```bash
python src/train.py
```

### 预测图像

```bash
python src/predict/predict.py
```

### 使用TTA预测

```bash
python src/predict/predict_tta.py
```

## 现阶段情况

目前刚起步，目标为识别图像中物品类别
