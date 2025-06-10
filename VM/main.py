# import cv2
# from pyhdfs import HdfsClient,HdfsFileNotFoundException
# import numpy as np
#
# if __name__ == '__main__':
#     # 创建 HDFS 连接客户端
#     client = HdfsClient(hosts="hadoop101:9870", user_name="user")
#
#     # 打开图片
#     hdfs_img_path = "/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"
#     try:
#         response = client.open(hdfs_img_path)
#         # 将二进制流转化为图片
#         mat = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
#
#         # 展示图像
#         cv2.imshow("demo_img", mat)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     except HdfsFileNotFoundException as e:
#         print(f"File not found: {e}")


import pyhdfs


def download_image_from_hdfs(hdfs_path, local_path):
    # 创建 HDFS 连接客户端
    client = pyhdfs.HdfsClient(hosts="hadoop101:9870", user_name="user")

    try:
        # 从 HDFS 打开文件
        with client.open(hdfs_path) as hdfs_file:
            # 将 HDFS 文件内容读入本地文件
            with open(local_path, "wb") as local_file:
                local_file.write(hdfs_file.read())
        print(f"Image successfully downloaded to {local_path}")
    except pyhdfs.HdfsFileNotFoundException as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # HDFS 上的图片路径
    hdfs_img_path = (
        "/user/lhr/big_data/Cyclone_Wildfire_Flood_Earthquake_Database/Cyclone/0.jpg"
    )
    # 本地保存图片的路径
    local_img_path = "/home/lhr/big_data/0.jpg"

    # 调用函数下载图片
    download_image_from_hdfs(hdfs_img_path, local_img_path)
