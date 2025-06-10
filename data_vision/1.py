# # -*- coding: utf-8 -*-
# # import pandas as pd
# # from pyecharts import options as opts
# # from pyecharts.charts import Pie
# #
# # def draw_pie_chart(data, html_file_path):
# #     # 创建饼状图
# #     pie_chart = (
# #         Pie()
# #         .add(
# #             series_name="地点统计",
# #             data_pair=data,
# #             radius=["40%", "75%"],  # 设置内外半径
# #             label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 设置标签格式
# #         )
# #         .set_global_opts(
# #             title_opts=opts.TitleOpts(title="地点出现次数的饼状图"),  # 设置标题
# #             legend_opts=opts.LegendOpts(orient="vertical", pos_left="left")  # 设置图例位置
# #         )
# #         .set_series_opts(
# #             tooltip_opts=opts.TooltipOpts(
# #                 is_show=True,
# #                 trigger="item",  # 悬停触发类型
# #                 formatter="{b}: {c} ({d}%)"  # 悬停时显示格式
# #             ),
# #             label_opts=opts.LabelOpts(
# #                 is_show=True,
# #                 position="outside",  # 标签显示位置
# #                 formatter="{b}: {d}%"  # 标签格式
# #             ),
# #             # label_line_opts=opts.LabelLineOpts(
# #             #     is_show=True,
# #             #     smooth=True  # 标签连接线是否平滑
# #             # ),
# #             # 动态特效
# #             emphasis_opts=opts.EmphasisOpts(
# #                 label_opts=opts.LabelOpts(
# #                     font_size=18,
# #                     font_weight="bold",
# #                     color="#FF6347"  # 悬停时字体颜色
# #                 ),
# #                 itemstyle_opts=opts.ItemStyleOpts(
# #                     color="rgba(255, 99, 71, 0.8)"  # 悬停时扇区颜色
# #                 )
# #             )
# #         )
# #     )
# #
# #     # 保存到 HTML 文件
# #     pie_chart.render(html_file_path)
# #     print(f"Pie chart saved to {html_file_path}")
# #
# # # 你提供的 data 内容
# # data = [
# #     ['中国四川省', 548], ['中国新疆维吾尔自治区南部', 471], ['中国西藏自治区', 352],
# #     ['中国台湾', 323], ['中国台湾地区', 263], ['中国云南省', 248],
# #     ['中国新疆维吾尔自治区北部', 241], ['塔吉克斯坦-中国新疆维吾尔自治区边境地区', 219],
# #     ['中国青海省', 152], ['中国东北部', 142], ['缅甸-中国边境地区', 129],
# #     ['吉尔吉斯斯坦-中国新疆维吾尔自治区边境地区', 112], ['中国甘肃省', 106],
# #     ['中国东南部', 92], ['中国内蒙古自治区西部', 60], ['中国东南沿海', 38],
# #     ['哈萨克斯坦-中国新疆维吾尔自治区边境地区', 37], ['中国西藏自治区-印度边境地区', 34],
# #     ['克什米尔-中国西藏自治区边境地区', 29], ['中国台湾东北', 25],
# #     ['克什米尔-中国新疆维吾尔自治区边境地区', 24], ['中国东海', 23],
# #     ['俄罗斯东部-中国东北边境地区', 21], ['中国黄海', 12],
# #     ['中国西藏自治区西部-印度边境地区', 8], ['中国台湾东南', 5], ['中国南海', 1]
# # ]
# # df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')
# # # 统计特定列元素的出现次数
# # column_name = '地点'  # 替换为你实际的列名
# # counts = df[column_name].value_counts()
# #
# # # 准备数据
# # data = [list(z) for z in zip(counts.index, counts.values)]
# #
# # # 调用函数，传入数据和 HTML 文件路径
# # html_file_path = './pie_chart.html'  # 替换为保存的 HTML 文件路径
# # draw_pie_chart(data, html_file_path)
#
# from pyecharts import options as opts
# from pyecharts.charts import Pie
# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv', encoding='utf-8')
#
# # 统计特定列元素的出现次数
# column_name = '地点'  # 替换为你实际的列名
# counts = df[column_name].value_counts()
#
# # 准备数据
# data = counts.reset_index().values.tolist()
# data = [[item[0], item[1]] for item in data]
# print(data)
# # data1 = [['中国四川省', 548], ['中国新疆维吾尔自治区南部', 471], ['中国西藏自治区', 352], ['中国台湾', 323], ['中国台湾地区', 263], ['中国云南省', 248], ['中国新疆维吾尔自治区北部', 241], ['塔吉克斯坦-中国新疆维吾尔自治区边境地区', 219], ['中国青海省', 152], ['中国东北部', 142], ['缅甸-中国边境地区', 129], ['吉尔吉斯斯坦-中国新疆维吾尔自治区边境地区', 112], ['中国甘肃省', 106], ['中国东南部', 92], ['中国内蒙古自治区西部', 60], ['中国东南沿海', 38], ['哈萨克斯坦-中国新疆维吾尔自治区边境地区', 37], ['中国西藏自治区-印度边境地区', 34], ['克什米尔-中国西藏自治区边境地区', 29], ['中国台湾东北', 25], ['克什米尔-中国新疆维吾尔自治区边境地区', 24], ['中国东海', 23], ['俄罗斯东部-中国东北边境地区', 21], ['中国黄海', 12], ['中国西藏自治区西部-印度边境地区', 8], ['中国台湾东南', 5], ['中国南海', 1]]
# # 创建饼状图
# pie_chart = (
#     Pie()
#     .add(
#         series_name="地点统计",
#         data_pair=data,
#         radius=["20%", "45%"],  # 设置内外半径
#         label_opts=opts.LabelOpts(formatter="{b}: {d}%")  # 设置标签格式
#     )
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="地点出现次数的饼状图",pos_left="right",  pos_top="top"),  # 设置标题
#         legend_opts=opts.LegendOpts(orient="vertical", pos_left="left")  # 设置图例位置
#     )
#     .set_series_opts(
#         tooltip_opts=opts.TooltipOpts(
#             is_show=True,
#             trigger="item",  # 悬停触发类型
#             formatter="{b}: {c} ({d}%)"  # 悬停时显示格式
#         ),
#         label_opts=opts.LabelOpts(
#             is_show=True,
#             position="outside",  # 标签显示位置
#             formatter="{b}: {d}%"  # 标签格式
#         ),
#         center=["70%", "50%"],
#         emphasis_opts=opts.EmphasisOpts(
#             label_opts=opts.LabelOpts(
#                 font_size=18,
#                 font_weight="bold",
#                 color="#FF6347"  # 悬停时字体颜色
#             ),
#             itemstyle_opts=opts.ItemStyleOpts(
#                 color="rgba(255, 99, 71, 0.8)"  # 悬停时扇区颜色
#             )
#         )
#     )
# )
#
# # 渲染到 HTML 文件
# html_file_path = './地点统计饼状图.html'
# pie_chart.render(html_file_path)
# print(f"Pie chart saved to {html_file_path}")


import pandas as pd
from geopy.geocoders import GoogleV3
from geopy.exc import GeocoderTimedOut
import time
from pyecharts.charts import Map
from pyecharts import options as opts
from geopy.geocoders import Baidu
# import requests
#
# # 读取数据
# df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')  # 替换为实际的文件路径
#
# # 提取经纬度
# longitudes = df['经度(°)']
# latitudes = df['纬度(°)']
#
# # 设置 Baidu Map API 密钥
# API_KEY = 'JmzD8rdWffZNVg7SmgEWYU5DB6BEsiKX'
#
# def get_province_from_coords(lat, lon):
#     url = f"https://api.map.baidu.com/reverse_geocoding/v3/?ak={API_KEY}&output=json&coordtype=gcj02&location={lat},{lon}"
#     response = requests.get(url)
#     # print(response)
#     if response.status_code == 200:
#         result = response.json()
#         print(result)
#         if result['status'] == 0:  # 请求成功
#             address_component = result['result']['addressComponent']
#             province = address_component.get('province', '未知省份')
#             return province
#         else:
#             print(f"Error in API response: {result['status']}")
#     return '无法获取'
#
# # 获取省份
# provinces = [get_province_from_coords(lat, lon) for lon, lat in zip(longitudes, latitudes)]
#
# # 统计省份出现次数
# province_counts = pd.Series(provinces).value_counts()
# # 准备数据
# data = province_counts.reset_index().values.tolist()
# data = [[item[0], item[1]] for item in data]
#
#
# # 绘制地图
# from pyecharts.charts import Map
# from pyecharts import options as opts
#
# # 绘制地图
# map_chart = (
#     Map()
#     .add(
#         series_name="省份出现次数",
#         data_pair=data,
#         maptype="china",  # 需要绘制中国地图
#         is_map_symbol_show=True  # 不显示地图标记
#     )
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="省份出现次数的地图", pos_left="center"),
#         visualmap_opts=opts.VisualMapOpts(
#             min_=0,
#             max_=province_counts.max(),
#             is_calculable=True,
#             dimension=0,
#             range_text=['高', '低']
#         ),
#         tooltip_opts=opts.TooltipOpts(
#             is_show=True,
#             trigger="item",  # 悬浮触发类型
#             formatter="{b}: {c} 次"  # 悬浮显示格式
#         ),
#     )
#     .set_series_opts(
#         emphasis_opts=opts.EmphasisOpts(
#             label_opts=opts.LabelOpts(
#                 font_size=18,
#                 font_weight="bold",
#                 color="#FF6347"  # 悬浮时字体颜色
#             ),
#             itemstyle_opts=opts.ItemStyleOpts(
#                 color="rgba(255, 99, 71, 0.8)"  # 悬浮时扇区颜色
#             )
#         )
#     )
# )
#
# # 保存到 HTML 文件
# map_chart.render('./province_map.html')
# print("地图已保存为 province_map.html")
# import pandas as pd
# from pyecharts import options as opts
# from pyecharts.charts import Map
#
# # 读取经纬度范围数据
# province_bounds = {
#     "安徽省": {"lon": (114.878463, 119.645188), "lat": (29.395191, 34.65234)},
#     "澳门特别行政区": {"lon": (113.528164, 113.598861), "lat": (22.109142, 22.217034)},
#     "北京市": {"lon": (115.416827, 117.508251), "lat": (39.442078, 41.058964)},
#     "福建省": {"lon": (115.84634, 120.722095), "lat": (23.500683, 28.317231)},
#     "甘肃省": {"lon": (92.337827, 108.709007), "lat": (32.596328, 42.794532)},
#     "广东省": {"lon": (109.664816, 117.303484), "lat": (20.223273, 25.519951)},
#     "广西壮族自治区": {"lon": (104.446538, 112.05675), "lat": (20.902306, 26.388528)},
#     "贵州省": {"lon": (103.599417, 109.556069), "lat": (24.620914, 29.224344)},
#     "海南省": {"lon": (108.614575, 117.842823), "lat": (8.30204, 20.16146)},
#     "河北省": {"lon": (113.454863, 119.84879), "lat": (36.048718, 42.615453)},
#     "河南省": {"lon": (110.35571, 116.644831), "lat": (31.3844, 36.366508)},
#     "黑龙江省": {"lon": (121.183134, 135.088511), "lat": (43.422993, 53.560901)},
#     "湖北省": {"lon": (108.362545, 116.132865), "lat": (29.032769, 33.272876)},
#     "湖南省": {"lon": (108.786106, 114.256514), "lat": (24.643089, 30.1287)},
#     "吉林省": {"lon": (121.638964, 131.309886), "lat": (40.864207, 46.302152)},
#     "江苏省": {"lon": (116.355183, 121.927472), "lat": (30.76028, 35.127197)},
#     "江西省": {"lon": (89.551219, 124.57284), "lat": (8.972204, 40.256391)},
#     "辽宁省": {"lon": (118.839668, 125.785614), "lat": (38.72154, 43.488548)},
#     "内蒙古自治区": {"lon": (97.17172, 126.065581), "lat": (37.406647, 53.333779)},
#     "宁夏回族自治区": {"lon": (104.284332, 107.661713), "lat": (35.238497, 39.387783)},
#     "青海省": {"lon": (89.401764, 103.068897), "lat": (31.600668, 39.212599)},
#     "山东省": {"lon": (114.810126, 122.705605), "lat": (34.377357, 38.399928)},
#     "山西省": {"lon": (110.230241, 114.56294), "lat": (34.583784, 40.744953)},
#     "陕西省": {"lon": (105.488313, 111.241907), "lat": (31.706862, 39.582532)},
#     "上海市": {"lon": (120.852326, 122.118227), "lat": (30.691701, 31.874634)},
#     "四川省": {"lon": (97.347493, 108.54257), "lat": (26.048207, 34.315239)},
#     "台湾省": {"lon": (119.314417, 123.701571), "lat": (21.896939, 25.938831)},
#     "天津市": {"lon": (116.702073, 118.059209), "lat": (38.554824, 40.251765)},
#     "西藏自治区": {"lon": (78.386053, 99.115351), "lat": (26.853453, 36.484529)},
#     "香港特别行政区": {"lon": (113.815684, 114.499703), "lat": (22.134935, 22.566546)},
#     "新疆维吾尔自治区": {"lon": (73.501142, 96.384783), "lat": (34.336146, 49.183097)},
#     "云南省": {"lon": (97.527278, 106.196958), "lat": (21.142312, 29.225286)},
#     "浙江省": {"lon": (118.022574, 122.834203), "lat": (26.643625, 31.182556)},
#     "重庆市": {"lon": (105.289838, 110.195637), "lat": (28.164706, 32.204171)},
# }
#
# # 读取经纬度数据
# df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv', encoding='utf-8')
#
# # 准备数据
# def get_province(lat, lon):
#     for province, bounds in province_bounds.items():
#         if bounds["lon"][0] <= lon <= bounds["lon"][1] and bounds["lat"][0] <= lat <= bounds["lat"][1]:
#             return province
#     return None
#
# # 使用经纬度来推断省份
# df['省份'] = df.apply(lambda row: get_province(row['纬度(°)'], row['经度(°)']), axis=1)
#
# # 统计省份出现次数
# province_counts = pd.Series(df['省份']).value_counts()
# data = province_counts.reset_index().values.tolist()
# data = [[item[0], item[1]] for item in data if item[0] is not None]
#
# # 创建地图
# map_chart = (
#     Map()
#     .add(
#         series_name="地点出现次数",
#         data_pair=data,
#         maptype="china",  # 显示中国地图
#         is_map_symbol_show=False  # 不显示地图上的标记点
#     )
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="地点出现次数地图", pos_left="center"),  # 设置标题及位置
#         visualmap_opts=opts.VisualMapOpts(
#             min_=0,
#             max_=province_counts.max(),
#             is_calculable=True,
#             orient="vertical",
#             pos_left="right",
#             pos_top="center"
#         )
#     )
#     .set_series_opts(
#         label_opts=opts.LabelOpts(is_show=True, formatter="{b}: {c}"),
#         emphasis_opts=opts.EmphasisOpts(
#             label_opts=opts.LabelOpts(
#                 font_size=18,
#                 font_weight="bold",
#                 color="#FF6347"  # 悬停时字体颜色
#             ),
#             itemstyle_opts=opts.ItemStyleOpts(
#                 color="rgba(255, 99, 71, 0.8)"  # 悬停时区域颜色
#             )
#         )
#     )
# )
#
# # 渲染图表
# map_chart.render("location_map_chart.html")
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Map
import json

# # 从 JSON 文件中读取经纬度范围数据
# with open('province_bounds.json', 'r', encoding='utf-8') as f:
#     province_bounds = json.load(f)
#
# # 读取经纬度数据
# df = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv', encoding='utf-8')
#
# # 准备数据
# def get_province(lat, lon):
#     for province, bounds in province_bounds.items():
#         if bounds["lon"][0] <= lon <= bounds["lon"][1] and bounds["lat"][0] <= lat <= bounds["lat"][1]:
#             return province
#     return None
#
# # 使用经纬度来推断省份
# df['省份'] = df.apply(lambda row: get_province(row['纬度(°)'], row['经度(°)']), axis=1)
# # 统计省份出现次数
# province_counts = pd.Series(df['省份']).value_counts()
#
# # 补充所有省份为 0 次出现数据
# all_provinces = list(province_bounds.keys())
# province_counts = province_counts.reindex(all_provinces, fill_value=0)
#
# # 统计省份出现次数
# data = province_counts.reset_index().values.tolist()
# data = [[item[0], item[1]] for item in data if item[0] is not None]
#
# # 创建地图
# map_chart = (
#     Map()
#     .add(
#         series_name="省份地震次数",
#         data_pair=data,
#         maptype="china",  # 显示中国地图
#         is_map_symbol_show=False  # 不显示地图上的标记点
#     )
#     .set_global_opts(
#         # title_opts=opts.TitleOpts(title="地点出现次数地图", pos_left="right"),  # 设置标题及位置
#         visualmap_opts=opts.VisualMapOpts(
#             min_=0,
#             max_=province_counts.max(),
#             is_calculable=True,
#             orient="vertical",
#             pos_left="right",
#             pos_top="center",
#             is_piecewise=True,  # 设置分段显示
#             pieces=[
#                 {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
#                 {"min": 1, "max": 10, "label": "1-10", "color": "#D4E157"},
#                 {"min": 11, "max": 50, "label": "11-50", "color": "#FFC107"},
#                 {"min": 51, "max": 100, "label": "51-100", "color": "#FF5722"},
#                 {"min": 101, "max": 1000, "label": "101+", "color": "#F44336"},
#             ]
#         )
#     )
#     .set_series_opts(
#         label_opts=opts.LabelOpts(is_show=True, formatter="{b}: {c}"),
#         emphasis_opts=opts.EmphasisOpts(
#             label_opts=opts.LabelOpts(
#                 font_size=18,
#                 font_weight="bold",
#                 color="#FF6347"  # 悬停时字体颜色
#             ),
#             itemstyle_opts=opts.ItemStyleOpts(
#                 color="rgba(255, 99, 71, 0.8)"  # 悬停时区域颜色
#             )
#         )
#     )
# )
#
# # 渲染图表
# map_chart.render("location_map_chart.html")

#
# import pandas as pd
# from collections import Counter
# from pyecharts import options as opts
# from pyecharts.charts import WordCloud
#
# # 读取 CSV 文件
# data = pd.read_csv('./processed_中国地震台网地震目录_1_normalized_MinMaxScaler.csv')
#
# # 提取地点数据
# locations = data['地点'].tolist()
#
# # 统计地点出现频率
# location_counts = Counter(locations)
#
# # 将统计结果转化为词云数据格式
# wordcloud_data = [{'name': location, 'value': count} for location, count in location_counts.items()]
# # 转换为元组列表
# wordcloud_data = [(item['name'], item['value']) for item in wordcloud_data]
# # 创建词云图对象
# wordcloud = WordCloud()
# # wordcloud_data =[{'name': '中国西藏自治区', 'value': 352}, {'name': '塔吉克斯坦-中国新疆维吾尔自治区边境地区', 'value': 219}, {'name': '中国东南部', 'value': 92}, {'name': '中国东北部', 'value': 142}, {'name': '中国新疆维吾尔自治区南部', 'value': 471}, {'name': '中国台湾地区', 'value': 263}, {'name': '中国台湾', 'value': 323}, {'name': '中国甘肃省', 'value': 106}, {'name': '中国新疆维吾尔自治区北部', 'value': 241}, {'name': '中国青海省', 'value': 152}, {'name': '中国西藏自治区-印度边境地区', 'value': 34}, {'name': '中国云南省', 'value': 248}, {'name': '吉尔吉斯斯坦-中国新疆维吾尔自治区边境地区', 'value': 112}, {'name': '中国台湾东北', 'value': 25}, {'name': '中国四川省', 'value': 548}, {'name': '中国台湾东南', 'value': 5}, {'name': '哈萨克斯坦-中国新疆维吾尔自治区边境地区', 'value': 37}, {'name': '中国西藏自治区西部-印度边境地区', 'value': 8}, {'name': '俄罗斯东部-中国东北边境地区', 'value': 21}, {'name': '缅甸-中国边境地区', 'value': 129}, {'name': '中国内蒙古自治区西部', 'value': 60}, {'name': '中国东海', 'value': 23}, {'name': '克什米尔-中国新疆维吾尔自治区边境地区', 'value': 24}, {'name': '克什米尔-中国西藏自治区边境地区', 'value': 29}, {'name': '中国黄海', 'value': 12}, {'name': '中国东南沿海', 'value': 38}, {'name': '中国南海', 'value': 1}]
# # 添加数据
# wordcloud.add(
#     series_name='地点词云',
#     data_pair=wordcloud_data,
#     word_size_range=[10, 40],  # 设置词云中词的大小范围
#     shape='star',  # 词云形状
#     # background_color='white'  # 背景颜色
# )
#
# # 设置全局配置
# wordcloud.set_global_opts(
#     title_opts=opts.TitleOpts(title="地点词云图"),
#     tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item"),  # 鼠标悬浮显示
#     visualmap_opts=opts.VisualMapOpts(max_=120, min_=15, is_show=True)  # 动态特效
# )
#
# # 渲染到本地 HTML 文件
# wordcloud.render('wordcloud.html')
# 读取省份边界数据

import pandas as pd
# import json
# from pyecharts import options as opts
# from pyecharts.charts import Map
# from PyQt5 import QtCore
# import os
#
# # 读取省份边界数据
# with open('./province_bounds.json', 'r', encoding='utf-8') as f:
#     province_bounds = json.load(f)
#
# # 读取地震数据
# df = pd.read_csv('./processed_地震灾情数据列表_scend.csv', encoding='utf-8')
#
# # 准备省份数据
# def get_province(lat, lon):
#     for province, bounds in province_bounds.items():
#         if bounds["lon"][0] <= lon <= bounds["lon"][1] and bounds["lat"][0] <= lat <= bounds["lat"][1]:
#             return province
#     return None
#
# # 使用经纬度推断省份
# df['省份'] = df.apply(lambda row: get_province(row['纬度'], row['经度']), axis=1)
#
# # 统计每个省份的死亡人数和经济损失
# province_stats = df.groupby('省份').agg(
#     {'死亡人数': 'sum', '直接经济损（万元）': 'sum'}).reset_index()
#
# # 补充所有省份为 0 次出现数据
# all_provinces = list(province_bounds.keys())
# province_stats = province_stats.set_index('省份').reindex(all_provinces, fill_value=0).reset_index()
#
# # 准备地图数据
# death_data = province_stats[['省份', '死亡人数']].values.tolist()
# economic_data = province_stats[['省份', '直接经济损（万元）']].values.tolist()
#
# # 创建地图
# map_chart = (
#     Map()
#     .add(
#         series_name="死亡人数",
#         data_pair=death_data,
#         maptype="china",  # 显示中国地图
#         is_map_symbol_show=False,
#         label_opts=opts.LabelOpts(is_show=True),  # 不显示标签
#     )
#     # .add(
#     #     series_name="经济损失",
#     #     data_pair=economic_data,
#     #     maptype="china",  # 显示中国地图
#     #     is_map_symbol_show=False,
#     #     label_opts=opts.LabelOpts(is_show=False),  # 不显示标签
#     # )
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="省份死亡人数"),
#         visualmap_opts=[
#             opts.VisualMapOpts(
#                 series_index=0,  # 针对死亡人数系列
#                 min_=0,
#                 max_=province_stats['死亡人数'].max(),
#                 is_calculable=True,
#                 orient="vertical",
#                 pos_left="right",
#                 pos_top="center",
#                 is_piecewise=True,
#                 pieces=[
#                     {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
#                     {"min": 1, "max": 10, "label": "1-10", "color": "#D4E157"},
#                     {"min": 11, "max": 50, "label": "11-50", "color": "#FFC107"},
#                     {"min": 51, "max": 100, "label": "51-100", "color": "#FF5722"},
#                     {"min": 101, "max": 9999999, "label": "101+", "color": "#F44336"},
#                 ]
#             ),
#             # opts.VisualMapOpts(
#             #     series_index=1,  # 针对经济损失系列
#             #     min_=0,
#             #     max_=province_stats['直接经济损（万元）'].max(),
#             #     is_calculable=True,
#             #     orient="vertical",
#             #     pos_left="left",
#             #     pos_top="center",
#             #     is_piecewise=True,
#             #     pieces=[
#             #         {"min": 0, "max": 0, "label": "0", "color": "#E0E0E0"},
#             #         {"min": 1, "max": 1000, "label": "1-1000万", "color": "#D4E157"},
#             #         {"min": 1001, "max": 10000, "label": "1001-10000万", "color": "#FFC107"},
#             #         {"min": 10001, "max": 50000, "label": "10001-50000万", "color": "#FF5722"},
#             #         {"min": 50001, "max": 100000, "label": "50001-100000万", "color": "#F44336"},
#             #     ]
#             # )
#         ]
#     )
#     .set_series_opts(
#         # tooltip_opts=opts.TooltipOpts(
#         #     is_show=True,
#         #     formatter=(
#         #         lambda params: (
#         #             f"{params.name}<br>死亡人数: {params.value[0]}<br>经济损失: {params.value[1]}万元"
#         #         )
#         #     )
#         # ),
#         emphasis_opts=opts.EmphasisOpts(
#             label_opts=opts.LabelOpts(
#                 font_size=18,
#                 font_weight="bold",
#                 color="#FF6347"  # 悬停时字体颜色
#             ),
#             itemstyle_opts=opts.ItemStyleOpts(
#                 color="rgba(255, 99, 71, 0.8)"  # 悬停时区域颜色
#             )
#         )
#     )
# )
#
# html_file_path = "./数据描述_地图.html"
#
# # 渲染图表
# map_chart.render(html_file_path)
# print(f"地图已保存到 {html_file_path}")
#
# # 在 PyQt5 中显示地图
# file_url = QtCore.QUrl.fromLocalFile(os.path.abspath(html_file_path))


# import pandas as pd
# from pyecharts import options as opts
# from pyecharts.charts import Bar
#
# # 读取CSV文件
# data = pd.read_csv('./processed_地震灾情数据列表_scend.csv')
#
# # 删除死亡人数为0的行
# data_cleaned = data[data['死亡人数'] != 0]
#
# # 提取数据
# dates = data_cleaned['发震时间(utc+8)']
# deaths = data_cleaned['死亡人数']
#
# # 创建柱状图
# bar = (
#     Bar()
#     .add_xaxis(dates.tolist())
#     .add_yaxis("死亡人数", deaths.tolist())
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="地震死亡人数柱状图"),
#         xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),  # x轴标签旋转
#         yaxis_opts=opts.AxisOpts(name="死亡人数"),
#         tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),  # 鼠标悬浮显示
#         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
#     )
# )
#
# # 渲染到本地 HTML 文件
# bar.render("earthquake_deaths_bar_chart_cleaned.html")
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Boxplot

# # 读取CSV文件
# data = pd.read_csv('./processed_地震灾情数据列表_scend.csv')
#
# # 删除任何包含NaN的行
# data_cleaned = data.dropna(subset=['震级(M)', '直接经济损（万元）', '死亡人数'])
#
# # 提取震级、经济损失和死亡人数数据
# magnitudes = data_cleaned['震级(M)'].tolist()
# economic_losses = data_cleaned['直接经济损（万元）'].tolist()
# deaths = data_cleaned['死亡人数'].tolist()
#
# # 为盒须图准备数据
# boxplot_data = [
#     magnitudes,
#     economic_losses,
#     deaths
# ]
#
# # 创建盒须图
# boxplot = (
#     Boxplot()
#     .add_xaxis(["震级", "经济损失", "死亡人数"])
#     .add_yaxis("数据分布", boxplot_data)
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="地震数据盒须图"),
#         yaxis_opts=opts.AxisOpts(name="数值"),
#         tooltip_opts=opts.TooltipOpts(
#             trigger="item",
#             axis_pointer_type="cross",
#             formatter="{b}<br/>最小值: {c[0]}<br/>下四分位数: {c[1]}<br/>中位数: {c[2]}<br/>上四分位数: {c[3]}<br/>最大值: {c[4]}"
#         ),
#         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
#     )
# )
#
# # 渲染到本地 HTML 文件
# boxplot.render("earthquake_data_boxplot.html")
# from pyspark.sql import SparkSession
# import pandas as pd
# from pyecharts import options as opts
# from pyecharts.charts import Line
#
# # Step 1: 创建 SparkSession
# spark = SparkSession.builder \
#     .appName("Read CSV with PySpark") \
#     .getOrCreate()
#
# # Step 2: 读取 CSV 文件
# df = spark.read.csv(r'D:\source\python\torch_big_data\data_vision\中国地震目录\dt_predictions.csv', header=True, inferSchema=True)
#
# # 显示数据（可选）
# df.show()
#
# # Step 3: 转换为 Pandas DataFrame
# pandas_df = df.toPandas()
#
# # 处理时间列，将其转换为字符串类型，以适应 pyecharts 的时间轴
# pandas_df['发震时刻(国际时)'] = pd.to_datetime(pandas_df['发震时刻(国际时)']).dt.strftime('%Y-%m-%d %H:%M:%S')
#
# # 提取数据
# x_data = pandas_df['发震时刻(国际时)'].tolist()  # 使用时间列作为 X 轴数据
# y_data_prediction = pandas_df['prediction'].tolist()  # 预测值
# y_data_ms = pandas_df['Ms'].tolist()  # Ms 值作为另一条线
#
# # Step 4: 使用 pyecharts 创建折线图
# line_chart = (
#     Line()
#     .add_xaxis(x_data)  # X 轴数据（时间）
#     .add_yaxis("预测值", y_data_prediction, label_opts=opts.LabelOpts(is_show=False))  # 添加预测值的折线，并隐藏数据点标签
#     .add_yaxis("Ms 值", y_data_ms, label_opts=opts.LabelOpts(is_show=False))  # 添加 Ms 值的折线，并隐藏数据点标签
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="Ms 和 Prediction 折线图"),
#         xaxis_opts=opts.AxisOpts(name="发震时刻(国际时)", type_="time"),  # X 轴为时间类型
#         # yaxis_opts=opts.AxisOpts(name="值"),
#         tooltip_opts=opts.TooltipOpts(
#             trigger="axis",
#             axis_pointer_type="cross",
#             # formatter=(
#             #     "<div>发震时刻: {b0}</div>"
#             #     "<div>预测值: {c[0]}</div>"
#             #     "<div>Ms 值: {c[1]}</div>"
#             # )
#         ),
#         datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")]  # 动态特效
#     )
# )
#
# # Step 5: 渲染到本地 HTML 文件
# line_chart.render("ms_vs_prediction_line_chart.html")
#
# # 关闭 SparkSession
# spark.stop()

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# import pandas as pd
# from pyecharts import options as opts
# from pyecharts.charts import HeatMap
#
# # 创建 SparkSession
# spark = SparkSession.builder.appName("HeatMapExample").getOrCreate()
#
# # 读取 CSV 文件
# df = spark.read.csv(r"D:\source\python\torch_big_data\data_vision\中国地震目录\lr_predictions.csv", header=True, inferSchema=True)
#
# # 选择震级和震源深度列
# df = df.select(col("Ms").alias("Magnitude"), col("震源深度(Km)").alias("Depth"))
#
# # 收集数据到本地
# data = df.toPandas()
#
# # 将数据转换为列表格式
# data_values = data[['Magnitude', 'Depth']].values.tolist()
#
# # 计算震级和震源深度的热力密度
# heatmap_data = []
# for magnitude in sorted(data['Magnitude'].unique()):
#     for depth in sorted(data['Depth'].unique()):
#         count = len(data[(data['Magnitude'] == magnitude) & (data['Depth'] == depth)])
#         heatmap_data.append([magnitude, depth, count])
#
# # 创建热力密度图
# heatmap = (
#     HeatMap()
#     .add_xaxis(sorted(data['Magnitude'].unique()))
#     .add_yaxis("Depth", sorted(data['Depth'].unique()), heatmap_data)
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="震级与震源深度热力密度图"),
#         xaxis_opts=opts.AxisOpts(name="震级"),
#         yaxis_opts=opts.AxisOpts(name="震源深度(Km)")
#     )
# )
#
# # 渲染图表
# heatmap.render("heatmap.html")
#
# print("热力密度图已保存为 heatmap.html")


from pyecharts import options as opts
from pyecharts.charts import Line

# # 训练损失数据
# train_losses = [
#     2.7587, 1.4161, 1.1737, 1.1362, 1.0965,
#     0.9885, 0.9133, 0.8075, 0.7512, 0.6856,
#     0.6317, 0.5827, 0.5537, 0.5053, 0.4990,
#     0.4732, 0.4433, 0.4221, 0.4032, 0.3971,
#     0.3641, 0.3592, 0.3699, 0.3503, 0.3586,
#     0.3613, 0.3015, 0.2773, 0.2805, 0.2500
# ]
#
# # 创建折线图
# line_chart = (
#     Line()
#     .add_xaxis(list(range(len(train_losses))))
#     .add_yaxis("训练损失", train_losses)
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="训练损失折线图"),
#         xaxis_opts=opts.AxisOpts(name="Epoch"),
#         yaxis_opts=opts.AxisOpts(name="损失值")
#     )
# )
#
# # 渲染图表
# line_chart.render("train_losses_line_chart.html")


# from pyecharts import options as opts
# from pyecharts.charts import Line
#
# # 验证损失值和准确率数据
# val_losses = [
#     2.5687, 1.2598, 1.1933, 1.1373, 1.1389,
#     1.0364, 0.9160, 0.7808, 0.7927, 1.0739,
#     0.7208, 0.5966, 0.8929, 0.6198, 0.5763,
#     0.6187, 0.5206, 0.5405, 0.4991, 0.5453,
#     0.5680, 0.4734, 0.4711, 0.5792, 0.6316,
#     0.4844, 0.4241, 0.4461, 0.4485, 0.4295
# ]
#
# val_accuracies = [
#     0.32, 0.33, 0.43, 0.43, 0.48,
#     0.48, 0.62, 0.67, 0.68, 0.56,
#     0.74, 0.81, 0.65, 0.79, 0.78,
#     0.76, 0.82, 0.78, 0.83, 0.81,
#     0.82, 0.85, 0.84, 0.83, 0.80,
#     0.83, 0.86, 0.87, 0.87, 0.86
# ]
#
# # 创建折线图
# line_chart = (
#     Line()
#     .add_xaxis(list(range(len(val_losses))))
#     .add_yaxis("验证损失", val_losses)
#     .add_yaxis("验证准确率", val_accuracies)
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="验证损失与准确率折线图"),
#         xaxis_opts=opts.AxisOpts(name="Epoch"),
#         # yaxis_opts=opts.AxisOpts(name="验证损失", position="left"),
#         # yaxis_opts=opts.AxisOpts(name="验证准确率", position="right", offset=80),
#         tooltip_opts=opts.TooltipOpts(trigger="axis")
#     )
# )
#
# # 渲染图表
# line_chart.render("val_metrics_line_chart.html")
#
# print("折线图已保存为 val_metrics_line_chart.html")
# from pyecharts import options as opts
# from pyecharts.charts import Line
#
# # 验证损失值和准确率数据
# val_losses = [
#     2.5687, 1.2598, 1.1933, 1.1373, 1.1389,
#     1.0364, 0.9160, 0.7808, 0.7927, 1.0739,
#     0.7208, 0.5966, 0.8929, 0.6198, 0.5763,
#     0.6187, 0.5206, 0.5405, 0.4991, 0.5453,
#     0.5680, 0.4734, 0.4711, 0.5792, 0.6316,
#     0.4844, 0.4241, 0.4461, 0.4485, 0.4295
# ]
#
# val_accuracies = [
#     0.32, 0.33, 0.43, 0.43, 0.48,
#     0.48, 0.62, 0.67, 0.68, 0.56,
#     0.74, 0.81, 0.65, 0.79, 0.78,
#     0.76, 0.82, 0.78, 0.83, 0.81,
#     0.82, 0.85, 0.84, 0.83, 0.80,
#     0.83, 0.86, 0.87, 0.87, 0.86
# ]
#
# # 创建折线图
# line_chart = (
#     Line()
#     .add_xaxis(list(range(len(val_losses))))
#     .add_yaxis("验证损失", val_losses, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2))
#     .add_yaxis("验证准确率", val_accuracies, is_smooth=True, linestyle_opts=opts.LineStyleOpts(width=2))
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="验证损失与准确率折线图"),
#         xaxis_opts=opts.AxisOpts(name="Epoch"),
#         yaxis_opts=opts.AxisOpts(name="值", position="left"),
#         tooltip_opts=opts.TooltipOpts(
#             trigger="axis",
#             axis_pointer_type="cross",
#             textstyle_opts=opts.TextStyleOpts(font_size=14)
#         ),
#         datazoom_opts=opts.DataZoomOpts()
#     )
# )
#
# # 渲染图表
# line_chart.render("val_metrics_line_chart.html")
#
# print("折线图已保存为 val_metrics_line_chart.html")
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import HeatMap

# 1. 读取数据
file_path = './深度学习模型/image_batch_results.txt'  # 替换为你的文件路径
df = pd.read_csv(file_path, delimiter='\s+', skiprows=1, names=['真实值', '预测值'])

# 获取唯一的标签
unique_labels_1 = df['真实值'].unique()  # 获取真实值的唯一标签
unique_labels_2 = df['预测值'].unique()  # 获取预测值的唯一标签
labels = pd.Index(unique_labels_1).union(pd.Index(unique_labels_2))  # 合并两个唯一标签列表

# 创建标签到索引的映射（使用循环和 i）
label_index = {}
for i in range(len(labels)):
    label_index[labels[i]] = i

# 初始化混淆矩阵
conf_matrix = pd.DataFrame(
    0,
    index=labels,
    columns=labels
)

# 使用 Pandas groupby 和 size 来统计数据
grouped = df.groupby(['真实值', '预测值']).size().reset_index(name='count')

# 填充混淆矩阵
for _, row in grouped.iterrows():
    true_label = row['真实值']
    pred_label = row['预测值']
    count = row['count']
    conf_matrix.at[true_label, pred_label] = count

# 2. 准备热力图数据
heatmap_data = []
for i in range(len(labels)):
    for j in range(len(labels)):
        heatmap_data.append([i, j, conf_matrix.iloc[i, j]])

# 将所有数据转换为整数类型
heatmap_data_int = [[int(cell) for cell in row] for row in heatmap_data]

# 3. 绘制混淆矩阵
heatmap = (
    HeatMap()
    .add_xaxis(labels.tolist())  # x轴标签
    .add_yaxis(
        "混淆矩阵",
        labels.tolist(),
        heatmap_data_int,
        label_opts=opts.LabelOpts(is_show=True, color='black'),  # 显示数据标签
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="混淆矩阵"),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            axislabel_opts=opts.LabelOpts(rotate=45)  # 旋转x轴标签
        ),
        yaxis_opts=opts.AxisOpts(type_="category"),
        visualmap_opts=opts.VisualMapOpts(),
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=True, color='black'),  # 显示数据标签
        itemstyle_opts=opts.ItemStyleOpts(
            border_color='#333',
            border_width=1
        )
    )
)

# 渲染到HTML文件
heatmap.render("confusion_matrix.html")
