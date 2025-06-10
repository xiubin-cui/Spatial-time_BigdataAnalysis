import matplotlib.pyplot as plt
from PIL import Image
import os

# 定义四个文件夹路径
folders = [
    r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Cyclone",
    r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Earthquake",
    r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Flood",
    r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Wildfire",
]

# 定义每个文件夹中要显示的图像数量
num_images_per_folder = 4

# 定义标签
labels = ["Cyclone", "Earthquake", "Flood", "Wildfire"]

# 创建一个 4x4 的画布
fig, axs = plt.subplots(4, 4, figsize=(15, 15))

# 读取并绘制图像
for i, folder in enumerate(folders):
    # 确保文件夹存在
    if not os.path.exists(folder):
        print(f"Folder {folder} does not exist.")
        continue

    # 获取文件夹中的图像文件列表
    images = [f for f in os.listdir(folder) if f.endswith(("png", "jpg", "jpeg"))]

    # 仅选择前 num_images_per_folder 张图像
    images = ["0.jpg", "1.jpg", "10.jpg", "101.jpg"]

    for j, image_file in enumerate(images):
        # 确保不会超出子图范围
        if i * num_images_per_folder + j >= 16:
            break

        image_path = os.path.join(folder, image_file)

        try:
            img = Image.open(image_path)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")  # 不显示坐标轴

            # 在每行的第一个图像上添加标签
            if j == 0:
                axs[i, j].text(
                    -0.1,
                    0.5,
                    labels[i],
                    fontsize=36,
                    verticalalignment="center",
                    horizontalalignment="right",
                    transform=axs[i, j].transAxes,
                )
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")

# 确保图像不会重叠
plt.tight_layout()
plt.show()
