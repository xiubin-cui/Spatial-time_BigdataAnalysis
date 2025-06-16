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
from typing import Optional, Tuple, List
from .config import LOGGING_CONFIG # 导入日志配置

# 配置日志
# 确保在 utils 模块中也正确配置了日志
# 使用 dictConfig 允许更复杂的配置，例如多个 handlers
if not logging.getLogger().handlers: # 避免重复配置
    handlers = []
    if "console" in LOGGING_CONFIG["handlers"]:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
        handlers.append(console_handler)
    if "file" in LOGGING_CONFIG["handlers"]:
        file_handler = logging.FileHandler(LOGGING_CONFIG["file_path"], encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOGGING_CONFIG["format"]))
        handlers.append(file_handler)

    logging.basicConfig(level=LOGGING_CONFIG["level"], handlers=handlers)

logger = logging.getLogger(__name__)


def safe_read_csv(
    file_path: str,
    delimiter: Optional[str] = None, # 默认为None，尝试自动推断
    skiprows: int = 0,
    names: Optional[list] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    安全地读取 CSV 文件，包含错误处理和日志记录。
    尝试多种编码和分隔符来增加健壮性。


    Args:
        file_path (str): 文件路径
        delimiter (str): 分隔符，默认为None，将尝试自动检测（逗号，然后是分号，然后是制表符，最后是空格）。
        skiprows (int): 跳过的行数，默认为 0
        names (Optional[list]): 列名列表，默认为 None。如果文件有头部，则设为 None。
        encoding (str): 文件编码，默认为 utf-8。

    Returns:
        pd.DataFrame: 读取的数据

    Raises:
        FileNotFoundError: 文件不存在
        pd.errors.EmptyDataError: 文件为空
        Exception: 其他意外错误
    """
    if not Path(file_path).exists():
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 尝试常用编码和分隔符
    encodings_to_try = [encoding, "gbk", "latin1"]
    delimiters_to_try = [',', ';', '\t', r'\s+'] if delimiter is None else [delimiter]

    for enc in encodings_to_try:
        for delim in delimiters_to_try:
            try:
                # header=0 表示第一行是头部，如果 names 不为 None 则覆盖此设置
                df = pd.read_csv(
                    file_path,
                    sep=delim,
                    skiprows=skiprows,
                    names=names,
                    encoding=enc,
                    header='infer' if names is None else None # 自动推断头部，除非 names 已提供
                )
                if df.empty and names is None: # 如果文件不为空但读取为空，可能是分隔符问题
                    logger.warning(f"文件 {file_path} 读取为空，尝试其他分隔符/编码。当前编码: {enc}, 分隔符: '{delim}'")
                    continue
                logger.info(f"成功以编码 '{enc}' 和分隔符 '{delim}' 读取文件: {file_path}")
                return df
            except pd.errors.EmptyDataError:
                logger.warning(f"文件 {file_path} 为空数据。")
                raise pd.errors.EmptyDataError(f"文件 {file_path} 为空。")
            except Exception as e:
                logger.debug(f"尝试以编码 '{enc}' 和分隔符 '{delim}' 读取 {file_path} 失败: {e}")
                continue # 继续尝试其他组合
    
    # 如果所有尝试都失败
    logger.error(f"无法读取文件: {file_path}。尝试了所有常见编码和分隔符，请检查文件格式。")
    raise Exception(f"无法读取文件: {file_path}。请检查文件格式或手动指定编码和分隔符。")


def display_image(
    image_path: str, label_widget: QMessageBox, size: Tuple[int, int]
) -> None:
    """
    在 QLabel 控件中显示图片。

    Args:
        image_path (str): 图片文件路径
        label_widget (QMessageBox): 显示图片的 QLabel 控件
        size (Tuple[int, int]): QLabel 的尺寸 (width, height)
    """
    try:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        image = Image.open(image_path)
        image = image.convert("RGB") # 确保是RGB模式

        label_width, label_height = size
        img_width, img_height = image.size

        # 计算缩放比例，确保图片完全可见并保持比例

        scale = min(label_width / img_width, label_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        
        # 使用 Image.LANCZOS 进行高质量缩放
        image = image.resize((new_width, new_height), Image.LANCZOS)

        image_np = np.array(image)
        height, width, channels = image_np.shape
        bytes_per_line = 3 * width
        qimage = QImage(
            image_np.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        label_widget.setPixmap(pixmap)
        label_widget.setScaledContents(True) # 允许图片在标签内缩放
        logger.info(f"成功显示图像: {image_path}")

    except FileNotFoundError as e:
        logger.error(f"图片文件未找到: {e}")
        show_message("错误", str(e), "critical")
    except Exception as e:
        logger.error(f"处理图像时发生错误: {e}")
        show_message("错误", f"无法显示图像: {e}", "critical")


def show_message(title: str, message: str, icon_type: str = "information") -> None:
    """
    显示消息框。

    Args:
        title (str): 消息框标题
        message (str): 消息内容
        icon_type (str): 消息框图标类型 ("information", "warning", "critical")
    """
    msg_box = QMessageBox()
    msg_box.setWindowTitle(title)
    msg_box.setText(message)

    if icon_type == "information":
        msg_box.setIcon(QMessageBox.Information)
    elif icon_type == "warning":
        msg_box.setIcon(QMessageBox.Warning)
    elif icon_type == "critical":
        msg_box.setIcon(QMessageBox.Critical)
    else:
        msg_box.setIcon(QMessageBox.NoIcon) # Default to no icon for unknown types
        logger.warning(f"未知消息图标类型: {icon_type}。使用默认无图标。")

    msg_box.exec_()