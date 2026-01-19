from PIL import Image
import os


def vertical_concatenate_images(folder1, folder2, output_folder):
    """
    竖直拼接两个文件夹中的同名图片。

    参数:
        folder1 (str): 第一个文件夹路径。
        folder2 (str): 第二个文件夹路径。
        output_folder (str): 输出文件夹路径。
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取第一个文件夹中的所有文件名
    files = os.listdir(folder1)

    for file_name in files:
        # 构造两个文件的完整路径
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)

        # 检查文件是否存在于两个文件夹中
        if os.path.exists(path1) and os.path.exists(path2):
            try:
                # 打开两张图片
                img1 = Image.open(path1)
                img2 = Image.open(path2)

                # 获取图片宽度和高度
                width1, height1 = img1.size
                width2, height2 = img2.size

                # 确保两张图片宽度一致，否则调整宽度
                if width1 != width2:
                    img2 = img2.resize((width1, int(height2 * (width1 / width2))))

                # 创建一个新的空白图片，高度为两张图片高度之和
                new_height = height1 + img2.height
                new_image = Image.new('RGB', (width1, new_height))

                # 将两张图片粘贴到新图片中
                new_image.paste(img1, (0, 0))
                new_image.paste(img2, (0, height1))

                # 保存结果图片
                output_path = os.path.join(output_folder, file_name)
                new_image.save(output_path)
                print(f"Saved concatenated image to {output_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File {file_name} not found in both folders.")


# 示例用法
folder1 = "/home/f523/disk1/sxp/mmfewshot/work_dirs/meta-rcnn_r101_c4_8xb4_voc-split1_10shot-fine-tuning_ms/vis/VOC2007/JPEGImages"
folder2 = "/home/f523/disk1/sxp/mmfewshot/work_dirs/vfa_r101_c4_8xb4_voc-split1_10shot-fine-tuning/vis/VOC2007/JPEGImages"
output_folder = "/home/f523/disk1/sxp/mmfewshot/result"

vertical_concatenate_images(folder1, folder2, output_folder)