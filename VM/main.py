import pyhdfs
import os

def initialize_hdfs_client(hosts="hadoop101:9870", user_name="user"):
    """
    初始化 HDFS 客户端连接

    参数:
        hosts (str): HDFS NameNode 地址，默认为 "hadoop101:9870"
        user_name (str): HDFS 用户名，默认为 "user"

    返回:
        pyhdfs.HdfsClient: 初始化后的 HDFS 客户端对象
    """
    try:
        return pyhdfs.HdfsClient(hosts=hosts, user_name=user_name)
    except Exception as e:
        print(f"初始化 HDFS 客户端失败: {e}")
        raise

def download_image_from_hdfs(hdfs_path, local_path, client=None):
    """
    从 HDFS 下载图像文件到本地

    参数:
        hdfs_path (str): HDFS 上的图像文件路径
        local_path (str): 本地保存路径
        client (pyhdfs.HdfsClient, optional): HDFS 客户端对象，若为 None 则新建

    返回:
        bool: 下载成功返回 True，失败返回 False
    """
    try:
        # 如果未提供客户端，则初始化一个
        if client is None:
            client = initialize_hdfs_client()

        # 确保本地目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # 从 HDFS 读取文件并写入本地
        with client.open(hdfs_path) as hdfs_file:
            with open(local_path, "wb") as local_file:
                local_file.write(hdfs_file.read())
        print(f"图像成功下载至: {local_path}")
        return True
    except pyhdfs.HdfsFileNotFoundException as e:
        print(f"文件未找到: {e}")
        return False
    except Exception as e:
        print(f"下载图像时发生错误: {e}")
        return False

def main():
    """
    主函数：从 HDFS 下载指定图像文件
    """
    # 定义 HDFS 和本地路径
    hdfs_img_path = "/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"
    local_img_path = "/home/lhr/big_data/0.jpg"

    # 下载图像
    download_image_from_hdfs(hdfs_img_path, local_img_path)

if __name__ == "__main__":
    main()