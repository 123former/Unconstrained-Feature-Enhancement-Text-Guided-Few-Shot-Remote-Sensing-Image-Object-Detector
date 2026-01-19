import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = "/home/f523/disk1/sxp/mmfewshot/data/DIOR/Annotations"
files = os.listdir(path)

area_list = []
ratio_list = []


def file_extension(path):
    return os.path.splitext(path)[1]


classes = ['chimney']#'ship',
for xmlFile in tqdm(files, desc='Processing'):
    if not os.path.isdir(xmlFile):
        if file_extension(xmlFile) == '.xml':
            tree = et.parse(os.path.join(path, xmlFile))
            root = tree.getroot()
            filename = root.find('filename').text
            # print("--Filename is", xmlFile)

            for Object in root.findall('object'):
                name = Object.find('name').text
                if name not in classes:
                    continue
                bndbox = Object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text

                area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
                area_list.append(area)
                # print("Area is", area)

                # ratio = (int(ymax) - int(ymin)) / (int(xmax) - int(xmin))
                # ratio_list.append(ratio)
                # print("Ratio is", round(ratio,2))

square_array = np.array(area_list)
square_max = np.max(square_array)
square_min = np.min(square_array)
square_mean = np.mean(square_array)
square_var = np.var(square_array)
plt.figure(1)
plt.hist(square_array, 20)
plt.xlabel('Area in pixel')
plt.ylabel('Frequency of area')
plt.title('Area\n' \
          + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
          + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
          )
plt.savefig(
    '/home/f523/disk1/sxp/mmfewshot/work_dirs/cls-meta-rcnn_r101_c4_8xb4_voc-SPLIT1_20shot-fine-tuning_ngnn_plus/Area_chimney.png')
# ratio_array = np.array(ratio_list)
# ratio_max = np.max(ratio_array)
# ratio_min = np.min(ratio_array)
# ratio_mean = np.mean(ratio_array)
# ratio_var = np.var(ratio_array)
# plt.figure(2)
# plt.hist(ratio_array, 20)
# plt.xlabel('Ratio of length / width')
# plt.ylabel('Frequency of ratio')
# plt.title('Ratio\n' \
#           + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
#           + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
#           )
