from hdfs import InsecureClient
from pathlib import Path
import os
from typing import Optional

class HDFSUploader:
    """用于将本地文件夹上传到 HDFS 的工具类"""
    
    def __init__(self, hdfs_url: str, hdfs_user: str):
        """
        初始化 HDFS 客户端

        Args:
            hdfs_url (str): HDFS NameNode 的 URL
            hdfs_user (str): HDFS 用户名
        """
        try:
            self.hdfs_client = InsecureClient(hdfs_url, user=hdfs_user)
        except Exception as e:
            raise RuntimeError(f"初始化 HDFS 客户端失败: {e}")

    def upload_directory(self, local_dir: str, hdfs_target_dir: str) -> None:
        """
        将本地文件夹及其内容上传到 HDFS

        Args:
            local_dir (str): 本地文件夹路径
            hdfs_target_dir (str): HDFS 目标文件夹路径

        Raises:
            FileNotFoundError: 如果本地文件夹不存在
            RuntimeError: 如果文件上传失败
        """
        local_path = Path(local_dir)
        
        # 验证本地文件夹是否存在
        if not local_path.is_dir():
            raise FileNotFoundError(f"本地文件夹 {local_dir} 不存在")

        # 创建 HDFS 目标文件夹
        try:
            self.hdfs_client.makedirs(hdfs_target_dir)
        except Exception as e:
            raise RuntimeError(f"创建 HDFS 文件夹 {hdfs_target_dir} 失败: {e}")

        # 遍历本地文件夹并上传文件
        for local_file_path in local_path.rglob("*"):
            if local_file_path.is_file():
                # 计算 HDFS 目标路径
                relative_path = local_file_path.relative_to(local_path)
                hdfs_file_path = f"{hdfs_target_dir}/{relative_path.as_posix()}"

                # 上传文件
                try:
                    with local_file_path.open("rb") as local_file:
                        self.hdfs_client.write(hdfs_file_path, local_file)
                        print(f"文件 {local_file_path} 已上传至 HDFS: {hdfs_file_path}")
                except Exception as e:
                    print(f"上传文件 {local_file_path} 失败: {e}")
                    continue

        print(f"文件夹 {local_dir} 已上传至 HDFS: {hdfs_target_dir}")

def main():
    """主函数，执行 HDFS 文件上传"""
    try:
        # 配置参数
        hdfs_url = "http://hadoop101:9870"
        hdfs_user = "lhr"
        local_dir_path = "/home/lhr/big_data"
        hdfs_target_dir = "/user/lhr/big_data"

        # 初始化上传器并执行上传
        uploader = HDFSUploader(hdfs_url, hdfs_user)
        uploader.upload_directory(local_dir_path, hdfs_target_dir)
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()