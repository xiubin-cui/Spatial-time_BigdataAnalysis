根据您的操作和错误信息，问题仍然存在是因为 Python 无法找到 `data_vision` 模块。以下是逐步解决方案：

### 1. 确认项目目录结构
请确保您的项目结构如下（在 `/root/spatial-time_-bigdata-analysis/` 目录下执行）：
```bash
cd /root/spatial-time_-bigdata-analysis/
ls -l
```
应该能看到：
```
data_vision/
├── __init__.py
├── main_entry.py
└── Main_window_logic.py
```

### 2. 添加正确的项目根目录到 PYTHONPATH
您当前的路径设置不正确，应该添加项目**根目录**（包含 data_vision 的目录）：
```bash
# 移除之前错误的设置
sed -i '/export PYTHONPATH=/d' ~/.bashrc

# 添加项目根目录（注意是上级目录）
echo "export PYTHONPATH=\$PYTHONPATH:/root/spatial-time_-bigdata-analysis" >> ~/.bashrc
source ~/.bashrc
```

### 3. 验证路径设置
```bash
python3 -c "import sys; print('\n'.join(sys.path))" | grep "spatial-time"
```
应该显示您的项目根目录路径。

### 4. 确保存在 __init__.py
```bash
# 在 data_vision 目录下创建空文件（如果不存在）
touch /root/spatial-time_-bigdata-analysis/data_vision/__init__.py
```

### 5. 正确的运行方式
现在您应该：
```bash
cd /root/spatial-time_-bigdata-analysis/
python3 -m data_vision.main_entry
```
或者：
```bash
cd /root/spatial-time_-bigdata-analysis/data_vision/
python3 main_entry.py
```

### 6. 如果仍报错
可能是文件权限问题：
```bash
chmod -R +r /root/spatial-time_-bigdata-analysis/
```

### 关键检查点
1. 确认 `data_vision/__init__.py` 存在
2. 确认 `sys.path` 包含项目根目录（不是 data_vision 目录本身）
3. 确保您是从正确目录运行

### 替代解决方案（推荐）
使用开发模式安装：
```bash
cd /root/spatial-time_-bigdata-analysis/
pip install -e .
```
（需要项目根目录有 `setup.py` 文件）

请按照这些步骤操作后，再次尝试运行程序。如果仍有问题，请提供：
```bash
ls -l /root/spatial-time_-bigdata-analysis/
cat /root/spatial-time_-bigdata-analysis/data_vision/__init__.py 2>/dev/null || echo "无此文件"
```