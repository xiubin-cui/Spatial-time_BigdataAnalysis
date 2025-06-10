from hdfs import InsecureClient
import os

# HDFS 配置
hdfs_url = "http://hadoop101:9870"
hdfs_user = "lhr"
hdfs_client = InsecureClient(hdfs_url, user=hdfs_user)

# 本地文件夹和 HDFS 目标路径
local_dir_path = "/home/lhr/big_data"
hdfs_target_dir = "/user/lhr/big_data"

# 创建 HDFS 目标文件夹
hdfs_client.makedirs(hdfs_target_dir)

# 遍历本地文件夹并上传每个文件
for root, dirs, files in os.walk(local_dir_path):
    for file in files:
        local_file_path = os.path.join(root, file)
        hdfs_file_path = os.path.join(
            hdfs_target_dir, os.path.relpath(local_file_path, local_dir_path)
        )
        with open(local_file_path, "rb") as local_file:
            hdfs_client.write(hdfs_file_path, local_file)
            print(f"File {local_file_path} uploaded to HDFS at {hdfs_file_path}")

print(f"Directory {local_dir_path} uploaded to HDFS at {hdfs_target_dir}")
