# -*- coding: utf-8 -*-
"""
混淆矩阵热力图生成模块，用于可视化分类模型的预测结果。
"""

import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap
from pathlib import Path
from utils import safe_read_csv, logger
from config import BATCH_RESULT_PATH, CONFUSION_MATRIX_PATH
from typing import Tuple, List

def create_confusion_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List, List]:
    """
    创建混淆矩阵并准备热力图数据。

    Args:
        df (pd.DataFrame): 包含真实值和预测值的数据，需有 '真实值' 和 '预测值' 列

    Returns:
        tuple: 混淆矩阵、热力图数据、标签列表

    Raises:
        ValueError: 数据格式不正确
        Exception: 其他意外错误
    """
    try:
        if not {"真实值", "预测值"}.issubset(df.columns):
            raise ValueError("数据必须包含 '真实值' 和 '预测值' 列")

        labels = sorted(set(df["真实值"]).union(set(df["预测值"])))
        conf_matrix = pd.crosstab(
            df["真实值"], df["预测值"], rownames=["真实值"], colnames=["预测值"]
        )
        conf_matrix = conf_matrix.reindex(index=labels, columns=labels, fill_value=0)
        heatmap_data = [
            [i, j, int(conf_matrix.iloc[i, j])]
            for i in range(len(labels))
            for j in range(len(labels))
        ]
        logger.info("成功创建混淆矩阵")
        return conf_matrix, heatmap_data, labels
    except ValueError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"创建混淆矩阵时发生错误: {str(e)}")
        raise


def create_heatmap(
    conf_matrix: pd.DataFrame, heatmap_data: list, labels: list, output_path: str
) -> None:
    """
    创建并渲染混淆矩阵热力图。

    Args:
        conf_matrix (pd.DataFrame): 混淆矩阵
        heatmap_data (list): 热力图数据
        labels (list): 标签列表
        output_path (str): 输出 HTML 文件路径

    Raises:
        Exception: 渲染或保存失败
    """
    try:
        heatmap = (
            HeatMap()
            .add_xaxis(labels)
            .add_yaxis(
                series_name="混淆矩阵",
                y_axis=labels,
                value=heatmap_data,
                label_opts=opts.LabelOpts(
                    is_show=True, color="#000000", position="inside", font_size=12
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="混淆矩阵热力图", pos_left="center", pos_top="top"
                ),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    name="预测值",
                    axislabel_opts=opts.LabelOpts(rotate=45, font_size=10),
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="category",
                    name="真实值",
                    axislabel_opts=opts.LabelOpts(font_size=10),
                ),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=float(conf_matrix.values.max()) or 1,  # 防止 max 为 0
                    is_calculable=True,
                    orient="vertical",
                    pos_left="right",
                    pos_top="center",
                    range_color=["#E0E0E0", "#FF6347"],
                ),
                tooltip_opts=opts.TooltipOpts(
                    is_show=True, trigger="item", formatter="{b0}: {c}"
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
        logger.error(f"渲染热力图时发生错误: {str(e)}")
        raise


def main():
    """
    主函数，执行混淆矩阵热力图生成流程。

    Raises:
        Exception: 程序执行失败
    """
    try:
        df = safe_read_csv(BATCH_RESULT_PATH, names=["真实值", "预测值"])
        conf_matrix, heatmap_data, labels = create_confusion_matrix(df)
        create_heatmap(conf_matrix, heatmap_data, labels, str(CONFUSION_MATRIX_PATH))
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()