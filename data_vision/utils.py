# -*- coding: utf-8 -*-
"""
工具模块，提供公共功能，如文件操作、图像处理、日志记录等。
提高代码复用性和可维护性。
"""

import logging
import pandas as pd
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from pathlib import Path
from typing import Optional, Tuple
from .config import LOGGING_CONFIG

# 配置日志
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def safe_read_csv(
    file_path: str,
    delimiter: str = r"\s+",
    skiprows: int = 0,
    names: Optional[list] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    安全地读取 CSV 文件，包含错误处理和日志记录。

    Args:
        file_path (str): 文件路径
        delimiter (str): 分隔符，默认为空格
        skiprows (int): 跳过的行数，默认为 0
        names (Optional[list]): 列名列表，默认为 None
        encoding (str): 文件编码，默认为 utf-8

    Returns:
        pd.DataFrame: 读取的数据

    Raises:
        FileNotFoundError: 文件不存在
        pd.errors.EmptyDataError: 文件为空
        Exception: 其他意外错误
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件 {file_path} 不存在")

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            skiprows=skiprows,
            names=names,
            encoding=encoding,
        )

        if df.empty:
            raise pd.errors.EmptyDataError("数据文件为空")

        logger.info(f"成功加载数据文件: {file_path}")
        return df

    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"加载数据时发生错误: {str(e)}")
        raise


def display_image(image_path: str, label_widget, max_size: Tuple[int, int]) -> None:
    """
    加载并显示图像到指定的 QLabel 控件。

    Args:
        image_path (str): 图像文件路径
        label_widget: QLabel 控件
        max_size (Tuple[int, int]): 最大显示尺寸 (宽度, 高度)

    Raises:
        FileNotFoundError: 图像文件不存在
        Exception: 图像处理错误
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件 {image_path} 不存在")

        image = Image.open(image_path).convert("RGB")
        label_width, label_height = max_size
        img_width, img_height = image.size
        scale = min(label_width / img_width, label_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        image_np = np.array(image)
        height, width, channels = image_np.shape
        bytes_per_line = 3 * width
        qimage = QImage(
            image_np.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap)
        label_widget.setScaledContents(True)
        logger.info(f"成功显示图像: {image_path}")

    except FileNotFoundError as e:
        logger.error(str(e))
        QMessageBox.critical(None, "错误", str(e))
    except Exception as e:
        logger.error(f"处理图像时发生错误: {str(e)}")
        QMessageBox.critical(None, "错误", "无法显示图像")


def show_message(title: str, message: str, icon: str = "information") -> None:
    """
    显示消息框。

    Args:
        title (str): 消息框标题
        message (str): 消息内容
        icon (str): 图标类型 ("information", "warning", "critical")
    """
    msg_box = QMessageBox()
    if icon == "warning":
        msg_box.setIcon(QMessageBox.Warning)
    elif icon == "critical":
        msg_box.setIcon(QMessageBox.Critical)
    else:
        msg_box.setIcon(QMessageBox.Information)
    msg_box.setWindowTitle(title)
    msg_box.setText(message)
    msg_box.exec_()
