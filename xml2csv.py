import pandas as pd
import os

# train = pd.read_csv('annot_csv.csv')
# #print(train.head(10))
#
# data = pd.DataFrame()
# data['format'] = train['image_path']
# print(data['format'])
# # add xmin, ymin, xmax, ymax and class as per the format required
# for i in range(data.shape[0]):
#     data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['clas_name'][i]
#
# data.to_csv('ilsvrc_annot.txt', header=None, index=None, sep=' ')

"""
#for val in data['image_path']:
#    print(type(val))

data = pd.read_csv('irls_csv.csv')
print(data.head())

img_paths = []
for img in data['image_path']:
    #file_path = 'ILSVRC/Data/CLS-LOC/train/' + class_name + '/' + image_name + ".JPEG"
    #print(img)
    pass
    #img_paths.append(file_path)

#data['image_path'] = img_paths

print(data.head(10))
#data.to_csv('irls_csv.csv')


# add xmin, ymin, xmax, ymax and class as per the format required
for i in range(data.shape[0]):
    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class'][i]

data.to_csv('ilsvrc_annotate.txt', header=None, index=None, sep=' ')

"""


#print("No. of XML files : {}".format(len(os.listdir('Annotations'))))
import xml.etree.ElementTree as ET

filenames, cell_type = [], []
xmin, xmax, ymin, ymax = [], [], [], []

base_dir = 'ILSVRC/Annotations/CLS-LOC/train'
train_images_dir = 'ILSVRC/Data/CLS-LOC/train'

for directory in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, directory)):
        for xml_file in os.listdir(os.path.join(base_dir, directory)):
            print(xml_file)
#             img_file = xml_file.split('.')[0] + ".JPEG"
#             img_file_path = os.path.join(train_images_dir, directory, img_file)
#             file_path = os.path.join(base_dir, directory, xml_file)
#             tree = ET.parse(file_path)
#             root = tree.getroot()
#             for child in root:
#                 if child.tag == 'filename' :
#                     for object in root.iter('object'):
#                         for item in object.iter():
#                             if item.tag == 'name' :
#                                 cell_type.append(item.text)
#                                 filenames.append(img_file_path)
#                             if item.tag == 'xmin' : xmin.append(item.text)
#                             if item.tag == 'xmax' : xmax.append(item.text)
#                             if item.tag == 'ymin' : ymin.append(item.text)
#                             if item.tag == 'ymax' : ymax.append(item.text)
#
# data = {'image_path': filenames, 'clas_name':cell_type, 'xmin': xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax}
# submission = pd.DataFrame(data)
# print(submission.head())
#
# submission.to_csv('annot_csv.csv')
