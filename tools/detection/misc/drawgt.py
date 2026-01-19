import os
import cv2
import xml.etree.ElementTree as ET


def draw_bounding_boxes(image_path, xml_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)
    # 解析xml文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 遍历object标签
    for obj in root.findall('object'):
        # 获取类别名称和bounding box坐标
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 在图像上绘制矩形框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 设置文字位置，并在图像上添加类别名称
        text_position = (xmin, ymin - 10) if ymin > 20 else (xmin, ymax + 15)
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存结果图片
    cv2.imwrite(output_path, image)


def main():
    annotations_dir = './data/DIOR/Annotations'  # VOC格式的XML标签文件存放目录
    images_dir = './data/DIOR/JPEGImages-test'  # 图片存放目录
    output_dir = './data/DIOR/gt'  # 结果存放目录

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历Annotations目录下的所有xml文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            # 获取对应的图片路径
            img_name = xml_file.replace('.xml', '.jpg')  # 假设图片都是.jpg格式
            img_path = os.path.join(images_dir, img_name)

            # 确认图片存在
            if os.path.exists(img_path):
                # 处理并保存结果
                draw_bounding_boxes(img_path, os.path.join(annotations_dir, xml_file),
                                    os.path.join(output_dir, img_name))
            else:
                print(f"Image {img_name} does not exist.")


if __name__ == '__main__':
    main()