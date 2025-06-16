# -*- coding: utf-8 -*-
"""
混淆矩阵热力图生成模块，用于可视化分类模型的预测结果。
"""

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pathlib import Path
from .utils import safe_read_csv, logger
from .config import BATCH_RESULT_PATH, CONFUSION_MATRIX_PATH, IMAGE_CLASS_NAMES # 导入 IMAGE_CLASS_NAMES
from typing import Tuple, List
import os

def create_confusion_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List, List]:
    """
    创建混淆矩阵并准备热力图数据。

    Args:
        df (pd.DataFrame): 包含真实值和预测值的数据，需有 '真实标签(推断)' 和 '预测值' 列

    Returns:
        tuple: 混淆矩阵、热力图数据、标签列表

    Raises:
        ValueError: 数据格式不正确
        Exception: 其他意外错误
    """
    try:
        # 确保列名与 Main_window_logic.py 中的一致
        required_cols = {"真实标签(推断)", "预测值"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"数据必须包含 {', '.join(missing)} 列。实际列: {df.columns.tolist()}")

        # 获取所有可能的标签，包括那些可能没有出现在当前数据中的类别（例如，某个类别没有被预测或作为真实值）
        # 使用 config 中定义的 IMAGE_CLASS_NAMES 确保标签的完整性
        all_labels = sorted(list(set(df["真实标签(推断)"]).union(set(df["预测值"])).union(set(IMAGE_CLASS_NAMES))))
        
        conf_matrix = pd.crosstab(
            df["真实标签(推断)"], df["预测值"], rownames=["真实值"], colnames=["预测值"]
        )
        
        # 重新索引混淆矩阵，确保所有类别都包含在内，缺失的用0填充
        conf_matrix = conf_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

        # 准备热力图数据
        data = []
        for i, row_label in enumerate(all_labels):
            for j, col_label in enumerate(all_labels):
                value = conf_matrix.loc[row_label, col_label]
                data.append([j, i, value]) # pyecharts 的热力图是 (x, y, value) 格式

        logger.info("成功创建混淆矩阵并准备热力图数据。")
        return conf_matrix, data, all_labels
    except Exception as e:
        logger.error(f"创建混淆矩阵失败: {e}", exc_info=True)
        raise

def create_heatmap(
    conf_matrix: pd.DataFrame, data: List, labels: List, output_path: str
) -> None:
    """
    生成混淆矩阵热力图并保存为 HTML 文件。

    Args:
        conf_matrix (pd.DataFrame): 混淆矩阵数据框
        data (List): 热力图数据列表
        labels (List): 类别标签列表
        output_path (str): HTML 输出文件路径
    """
    try:
        heatmap = (
            HeatMap()
            .add_xaxis(labels)
            .add_yaxis(
                "预测结果",
                labels, # Y轴标签
                data,
                label_opts=opts.LabelOpts(is_show=True, position="inside", formatter="{c}"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="混淆矩阵热力图"),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    name="预测值",
                    axislabel_opts=opts.LabelOpts(font_size=10),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="category",
                    name="真实值",
                    axislabel_opts=opts.LabelOpts(font_size=10),
                ),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=float(conf_matrix.values.max()) if not conf_matrix.empty else 1,  # 防止 max 为 0 或空
                    is_calculable=True,
                    orient="vertical",
                    pos_left="right",
                    pos_top="center",
                    range_color=["#E0E0E0", "#FF6347"], # 颜色范围从浅灰到番茄红
                ),
                tooltip_opts=opts.TooltipOpts(
                    is_show=True, trigger="item", formatter="真实值: {b1}<br/>预测值: {b0}<br/>数量: {c}" # 改进提示信息
                ),
            )
            .set_series_opts(
                itemstyle_opts=opts.ItemStyleOpts(
                    border_color="#333333", border_width=1
                )
            )
        )
        heatmap.render(output_path)
        logger.info(f"热力图已保存至: {output_path}")
    except Exception as e:
        logger.error(f"渲染热力图时发生错误: {e}", exc_info=True)
        raise


def main():
    """
    主函数，执行混淆矩阵热力图生成流程。

    Raises:
        Exception: 程序执行失败
    """
    try:
        # BATCH_RESULT_PATH 现在是 CSV 文件，并且应该有头部，所以不需要 names 参数
        df = safe_read_csv(BATCH_RESULT_PATH) 
        conf_matrix, heatmap_data, labels = create_confusion_matrix(df)
        create_heatmap(conf_matrix, heatmap_data, labels, str(CONFUSION_MATRIX_PATH))
        print(f"混淆矩阵热力图已生成并保存到: {CONFUSION_MATRIX_PATH}")
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except pd.errors.EmptyDataError:
        print("警告: 输入数据文件为空。")
    except ValueError as e:
        print(f"数据错误: {e}")
    except Exception as e:
        print(f"程序执行失败: {e}")
        logger.error(f"主函数执行失败: {e}", exc_info=True)


if __name__ == "__main__":
    main()