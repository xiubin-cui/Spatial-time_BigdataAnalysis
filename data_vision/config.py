# -*- coding: utf-8 -*-
"""
配置文件，定义项目中使用的路径、模型参数和其他常量。
提高代码的可维护性和扩展性，避免硬编码。
"""

import os
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 数据路径
DATA_DIR = BASE_DIR / "source"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "output"

# 确保输出目录存在
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 模型和数据文件路径
# 修正：确保路径指向正确的文件类型和位置
IMAGE_MODEL_PATH = MODEL_DIR / "resnet18_0.01_source.pth" # 假设模型文件名为此
BATCH_RESULT_PATH = OUTPUT_DIR / "image_batch_results.csv" # 修正：改为CSV文件，存储预测结果
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.html"

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": [
        "console",  # Console handler
        "file"      # File handler
    ],
    "file_path": BASE_DIR / "app.log" # Log file path
}

# 支持的文本类型和其对应的数据集配置
# 将数据集配置从 Main_window_logic.py 移到这里，方便管理和扩展
DATASET_CONFIGS = {
    "中国地震台网地震目录数据集训练": {
        "file_path": DATA_DIR / "中国地震台网地震目录数据集训练.csv", # 假设文件路径
        "feature_columns": ["震源深度(Km)", "mL", "mb", "mB"],
        "label_column": "Ms7",
        "description": "中国地震台网数据"
    },
    "全球地震台网地震目录数据集训练": {
        "file_path": DATA_DIR / "全球地震台网地震目录数据集训练.csv", # 假设文件路径
        "feature_columns": ["震源深度(Km)", "Ms7", "mL", "mb", "mB"],
        "label_column": "Ms",
        "description": "全球地震台网数据"
    },
    "强震动参数数据集训练": {
        "file_path": DATA_DIR / "强震动参数数据集训练.csv", # 假设文件路径
        "feature_columns": [
            "震源深度", "震中距", "仪器烈度", "总峰值加速度PGA", "总峰值速度PGV",
            "参考Vs30", "东西分量PGA", "南北分量PGA", "竖向分量PGA",
            "东西分量PGV", "南北分量PGV", "竖向分量PGV"
        ],
        "label_column": "震级",
        "description": "强震动参数数据"
    }
}

# 支持的文本类型 (从 DATASET_CONFIGS 动态生成)
TEXT_TYPES = list(DATASET_CONFIGS.keys())

# 支持的可视化类型
VISUALIZATION_TYPES = [
    "数据描述",
    "深度学习模型",
    "线性回归模型",
    "决策树回归模型",
    "随机森林回归模型",
    "梯度提升回归树模型",
]

# 可视化图表类型与可视化模型的映射
VISUALIZATION_CHARTS = {
    "数据描述": ["散点图", "直方图", "箱线图", "折线图", "柱状图"],
    "深度学习模型": ["混淆矩阵图", "ROC曲线"],
    "线性回归模型": ["残差图", "预测对比图"],
    "决策树回归模型": ["树结构图"],
    "随机森林回归模型": ["特征重要性图"],
    "梯度提升回归树模型": ["特征重要性图", "残差图"],
}

# 图像分类类别 (与图像模型训练时的类别一致)
IMAGE_CLASS_NAMES = ["Cyclone", "Earthquake", "Flood", "Wildfire"] # 假设这些是您的类别