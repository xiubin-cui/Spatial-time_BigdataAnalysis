### 项目介绍

**项目概述：**

该项目名为“针对自然灾害图像及文本分析”，旨在利用大数据和机器学习技术分析和预测与自然灾害（飓风、地震、洪水和野火）相关的数据。主要目标包括：

- 数据存储与处理： 采用 Hadoop 分布式文件系统（HDFS）进行分布式存储，使用 Apache Spark 进行高效数据处理，包括文本数据的预处理（清洗、分词、特征提取）和图像数据的预处理（归一化、调整大小、去噪）使用 OpenCV。
- 深度学习图像分类： 实现 ResNet-18 模型对自然灾害图像进行高精度分类，通过数据增强技术（如归一化、旋转、颜色抖动）提升模型的泛化能力。
- 机器学习文本预测： 使用 Spark 的 MLlib 训练模型（如线性回归、决策树回归、随机森林回归和梯度提升回归），对地震文本数据进行处理和分析，预测地震震级。
- 模型整合与应用： 结合图像分类和文本预测模型，探索其在自然灾害管理中的应用潜力，提供全面的灾害预警和应对策略。
- 数据可视化： 利用 PyQt 设计用户交互界面，结合 PyEcharts 实现数据可视化（如柱状图、折线图、热力地图、混淆矩阵等），直观展示模型预测结果和数据分析洞察。

实验数据包括 Kaggle 的 4428 张自然灾害图像数据集（1.84GB）和中国地震中心提供的多个地震相关文本数据集（约 20MB），涵盖地震目录、强震动参数和灾情数据。

实验流程包括环境搭建（HDFS、HBase、Spark）、数据采集与预处理、模型训练与评估、结果可视化以及系统集成。

**关键技术：**

- 数据存储： HDFS 用于分布式存储，Spark 用于数据处理。
- 模型训练: ResNet-18 用于图像分类，Spark MLlib 的多种回归模型用于文本预测。
- 可视化： PyQt 提供交互界面，PyEcharts 生成动态图表。
- 优化： 通过网格搜索和交叉验证优化模型参数，提升预测准确性和鲁棒性。

**成果：**

- 图像分类准确率达到 85%-91%，文本预测模型准确率在 70%以上（随机森林和决策树模型高达 80%以上）。
- 实现了直观的可视化界面，支持用户交互式分析和结果展示。

### 实验数据描述

数据集的图像数据来源于 Kaggle，数据集名称为“Cyclone, Wildfire, Flood and Earthquake”。此数据集包含了飓风、火灾、洪水和地震等自然灾害的图像，共计 4428 张图像，总数据量为 1.84 GB。数据下载链接为 [Kaggle 数据集](https://www.kaggle.com/datasets/aswin1871/cyclonewildfireflood-and-earthquake)。

文本数据则来源于中国地震中心，[下载链接](https://data.earthquake.cn/datashare/report.shtml?PAGEID=earthquake_csn)，涵盖多个地震相关数据集，包括：

1. 中国地震台网地震目录：记录了地震的详细信息，包含字段如发震时刻（国际时）、经度（°）、纬度（°）、震源深度（Km）、震级等。该数据集提供了中国地震台网记录的地震信息，用于地震事件的详细分析。
2. 全球地震台网地震目录：类似于中国地震台网地震目录，但记录的是全球范围内的地震信息。包含的字段包括序号、发震时刻（国际时）、经度（°）、纬度（°）、震源深度（Km）等。
3. 强震动参数数据集：包含了强震动参数的详细记录，如发震时间、震中纬度、震中经度、震源深度、震级、台网代码、台站名称等。还包括了地震的加速度和速度参数，这些参数对于理解地震对不同地区的影响至关重要。
4. 地震灾情数据列表：记录了地震的灾后信息，包括发震时间（UTC+8）、经度、纬度、震级、震源深度、震中位置、死亡人数以及直接经济损失（万元）等。

文本数据集的规模包括：中国地震台网地震目录（3716 行）、全球地震台网地震目录（28170 行）、强震动参数数据集（33960 行）和地震灾情数据列表（615 行），总计文本数据约 20 MB，所有数据均以 CSV 文件格式存储。

### 项目文件分析


### 项目结构分析

#### 1. 顶层结构

- `code/`：核心代码目录，包含数据处理、模型训练、预测、可视化等主要功能模块。
- `data_vision/`：主要为数据可视化和界面相关代码，含有大量数据处理和展示逻辑。
- `研究报告.docx`、`汇报PPT.pptx`：项目文档和汇报材料。

---

#### 2. code 目录结构与关系

- **数据处理与模型训练**
  - `image_processing.py`：图像数据预处理（如归一化、调整大小、去噪）。
  - `splite_data.py`：数据集划分。
  - `file_image_source.py`：图像文件源管理。
  - `base_achive.py`、`image_classification.py`、`image_prediction.py`：深度学习模型（如 ResNet-18）实现与推理。
  - `image_batch_results.txt`、`results.txt`：模型批量预测结果或评估结果。
- **VM 子目录**（主要为 PySpark 相关的数据处理与机器学习）
  - `text_data_preprocessing_1.py`、`text_data_preprocessing_2.py`、`image_data_processing.py`：文本和图像数据的 Spark 预处理。
  - `predict_processed_*.py`：针对不同地震数据集的机器学习预测脚本（如中国地震台网、全球地震台网、强震动参数）。
  - `upload_data_HDFS.py`：数据上传到 HDFS。
  - `read_data.py`、`spark_check.py`：数据读取与 Spark 环境检测。
  - `main.py`、`test.py`：主入口或测试脚本。
- **数据目录**
  - `data/Cyclone_Wildfire_Flood_Earthquake_Database/`：Kaggle 灾害图像数据集，按灾害类型（Cyclone、Wildfire、Flood、Earthquake）分类存储。

---

#### 3. data_vision 目录结构与关系

- **数据可视化与界面**
  - `wed_data.py`：核心数据处理与可视化脚本，集成了文件选择、数据读取（csv）、Spark 处理、模型预测、结果保存与展示等功能。
  - `data_view.py`、`data_view.ui`、`untitled.py`、`untitled.ui`：PyQt 界面与交互逻辑。
  - `data_file.py`、`qt.py`：界面相关的辅助脚本。
  - `1.py`、`untitled.py`：可能为实验性或临时脚本。
- **缓存与编译**
  - `__pycache__/`：Python 编译缓存。

---

#### 4. 文件间关系与数据流

**数据流动**

1. **原始数据**：图像数据存于 `code/data/Cyclone_Wildfire_Flood_Earthquake_Database/`，文本数据（如地震目录、灾情等）以 CSV 格式存储。
2. **预处理**：
   - 图像：`image_processing.py`、`code/VM/image_data_processing.py` 进行归一化、增强等处理。
   - 文本：`code/VM/text_data_preprocessing_1.py`、`text_data_preprocessing_2.py` 进行清洗、特征提取等。
3. **模型训练与预测**：
   - 图像：`base_achive.py`、`image_classification.py`、`image_prediction.py` 实现 ResNet-18 等深度学习模型。
   - 文本：`code/VM/predict_processed_*.py` 用 Spark MLlib 训练回归模型，预测地震震级等。
4. **结果输出**：预测结果保存为 CSV 或 TXT 文件（如 `results.txt`、`image_batch_results.txt`），供可视化模块读取。
5. **可视化与交互**：`data_vision/wed_data.py` 及相关 PyQt 脚本负责界面展示、结果可视化（如柱状图、热力图、混淆矩阵等）。

**文件引用与依赖**

- 许多脚本通过 `pandas.read_csv` 或 `spark.read.csv` 读取中间处理结果或预测结果。
- 结果文件（如 `*_predictions.csv`、`*_MinMaxScaler.csv`）在不同脚本间流转，用于后续分析或展示。

---

### 总结

- **code/** 负责数据处理、模型训练与预测，VM 子目录偏重于 Spark 分布式处理和机器学习，主目录偏重于深度学习图像处理。
- **data_vision/** 负责数据可视化和用户交互，集成了数据读取、Spark 处理、结果展示等功能。
- **数据流** 从原始数据（图像/文本）→ 预处理 → 模型训练/预测 → 结果保存 → 可视化展示，形成完整的分析闭环。
- **文档**（docx、pptx）为项目说明和成果展示。

如需进一步分析某一部分的具体实现或代码细节，请告知具体文件或功能点！

### 执行步骤

#### 步骤 1: 数据预处理

1. 图像数据预处理：

```bash
python code/VM/image_data_processing.py
```

2. 文本数据预处理：

```bash
python code/VM/text_data_preprocessing.py
```

#### 步骤 2: 模型训练

1. 图像分类模型训练（ResNet-18）：

```bash
python code/image_classification.py
```

2. 文本预测模型训练：

```bash
# 中国地震台网地震目录预测
python code/VM/predict_processed_中国地震台网地震目录.py

# 全球地震台网地震目录预测
python code/VM/predict_processed_全球地震台网地震目录.py

# 强震动参数数据集预测
python code/VM/predict_processed_强震动参数数据集.py
```

#### 步骤 3: 模型预测

1. 图像预测：

```bash
python code/image_prediction.py
```

#### 步骤 4: 可视化展示

启动可视化界面：

```bash
python code/data_vision/wed_data.py
```

### 4. 预期结果

- 图像分类准确率：85%-91%
- 文本预测模型准确率：70%以上（随机森林和决策树模型可达 80%以上）
- 可视化界面将展示：
  - 柱状图
  - 折线图
  - 热力地图
  - 混淆矩阵等

### 注意事项：

1. 确保所有依赖包都已正确安装
2. 数据文件需要放在正确的目录下
3. 如果使用 HDFS，需要确保 Hadoop 环境正确配置
4. 运行顺序建议按照上述步骤进行，因为后续步骤可能依赖于前面步骤的输出结果

如果您需要运行特定部分或者需要更详细的说明，请告诉我，我可以为您提供更具体的指导。
