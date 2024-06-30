import os
import pandas as pd


# def convert_labeled_list(csv_list, r=1):
#     img_pair_list = list()
#     for csv_file in csv_list:
#         with open(csv_file, 'r') as f:
#             img_in_csv = f.read().split('\n')[1:-1]
#         img_pair_list += img_in_csv
#     img_list = [i.split(',')[0] for i in img_pair_list]
#     if len(img_pair_list[0].split(',')) == 1:
#         label_list = None
#     else:
#         label_list = [i.split(',')[-1].replace('.tif', '-{}.tif'.format(r)) for i in img_pair_list]
#     return img_list, label_list

def convert_labeled_list(root, csv_list):
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        print(os.path.join(root, csv_file))
        data = pd.read_csv(os.path.join(root, csv_file))
        img_list += data['image'].tolist()
        label_list += data['mask'].tolist()
    return img_list, label_list