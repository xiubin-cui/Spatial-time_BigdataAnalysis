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
IMAGE_MODEL_PATH = MODEL_DIR / "fuquqi_base_model_18_0.1_nolaji_source.pth"  # BUG
BATCH_RESULT_PATH = MODEL_DIR / "image_batch_results.txt"  # BUG
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.html"  # BUG

# 支持的文本类型
TEXT_TYPES = ["中国地震目录", "全球地震目录", "强震动参数数据集", "地震灾情数据列表"]

# 支持的可视化类型
VISUALIZATION_TYPES = [
    "数据描述",
    "深度学习模型",
    "线性回归模型",
    "决策树回归模型",
    "随机森林回归模型",
    "梯度提升回归树模型",
]

# 可视化图表类型映射
VISUALIZATION_CHARTS = {
    "数据描述": ["饼状图", "柱状图", "地图", "盒须图", "词云图", "散点图", "折线图"],
    "深度学习模型": ["train折线图", "val折线图", "混淆矩阵图"],
    "线性回归模型": ["折线图", "混淆图"],
    "决策树回归模型": ["折线图", "混淆图"],
    "随机森林回归模型": ["折线图", "混淆图"],
    "梯度提升回归树模型": ["折线图", "混淆图"],
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "filename": OUTPUT_DIR / "app.log",
    "filemode": "a",  # 追加模式
}
