import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Tuple

def get_image_files(folder: Path, extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """
    获取指定文件夹中符合扩展名的图像文件列表。

    Args:
        folder (Path): 文件夹路径
        extensions (Tuple[str, ...]): 支持的图像文件扩展名，默认为 (".png", ".jpg", ".jpeg")

    Returns:
        List[Path]: 图像文件路径列表
    """
    return [f for f in folder.iterdir() if f.suffix.lower() in extensions]

def display_images(
    folders: List[str],
    labels: List[str],
    num_images_per_folder: int = 4,
    figsize: Tuple[int, int] = (15, 15)
) -> None:
    """
    显示多个文件夹中的图像，排列为网格。

    Args:
        folders (List[str]): 文件夹路径列表
        labels (List[str]): 每个文件夹对应的标签
        num_images_per_folder (int): 每个文件夹显示的图像数量，默认为4
        figsize (Tuple[int, int]): 画布大小，默认为 (15, 15)
    """
    # 创建4x4的画布
    fig, axs = plt.subplots(len(folders), num_images_per_folder, figsize=figsize)

    # 确保axs是二维数组，即使只有一个文件夹
    if len(folders) == 1:
        axs = [axs]

    for i, folder in enumerate(folders):
        folder_path = Path(folder)
        
        # 验证文件夹存在
        if not folder_path.is_dir():
            print(f"文件夹 {folder} 不存在")
            continue

        # 获取图像文件列表
        images = get_image_files(folder_path)
        
        # 使用固定的图像文件名
        selected_images = ["0.jpg", "1.jpg", "10.jpg", "101.jpg"]
        selected_images = [img for img in selected_images if (folder_path / img).is_file()]
        
        for j, image_file in enumerate(selected_images[:num_images_per_folder]):
            try:
                img_path = folder_path / image_file
                with Image.open(img_path) as img:
                    axs[i][j].imshow(img)
                    axs[i][j].axis("off")

                    # 在每行第一个图像添加标签
                    if j == 0:
                        axs[i][j].text(
                            -0.1,
                            0.5,
                            labels[i],
                            fontsize=36,
                            verticalalignment="center",
                            horizontalalignment="right",
                            transform=axs[i][j].transAxes,
                        )
            except Exception as e:
                print(f"打开图像 {img_path} 失败: {e}")
                continue

    # 调整布局防止重叠
    plt.tight_layout()
    plt.show()

def main():
    """主函数，设置参数并显示图像"""
    try:
        # 定义文件夹路径和标签
        folders = [
            r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Cyclone",
            r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Earthquake",
            r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Flood",
            r"D:\source\python\torch_big_data\Cyclone_Wildfire_Flood_Earthquake_Database\Wildfire",
        ]
        labels = ["Cyclone", "Earthquake", "Flood", "Wildfire"]
        num_images_per_folder = 4

        # 显示图像
        display_images(folders, labels, num_images_per_folder)
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()