import numpy as np
from PIL import Image
import os
import pickle
from PIL import Image
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import *

def get_data(root,csv_list,target_size=(384,384)):
    name = csv_list[0].split('.')[0]
    img_list = list()
    label_list = list()
    for csv_file in csv_list:
        print(os.path.join(root, csv_file))
        data_csv = pd.read_csv(os.path.join(root, csv_file))
        img_list = data_csv['image'].tolist()
        label_list = data_csv['mask'].tolist()

    images = []
    labels = []
    names = []
    for i in range(len(img_list)):
        img_file = join(root,img_list[i])
        label_file = join(root,label_list[i])
        img = Image.open(img_file)
        label = Image.open(label_file)

        img = img.resize(target_size)
        label = label.resize(target_size, resample=Image.NEAREST)

        img = np.array(img)

        if(len(img.shape)==3):
            img = img.transpose(2, 0, 1).astype(np.float32)
        # img = img.astype(np.uint8)
        label = np.array(label, dtype=np.int64)
        images.append(img)
        labels.append(label)
        names.append(img_list[i])

    with open(name+'.pkl','wb')as f:
        pickle.dump((images,labels,names),f)




















# def get_data(data_path):
#     img_paths = sorted(list(data_path.glob("*_img.png")))
#     mask_paths = sorted(list(data_path.glob("*_masks.png")))
#     images = []
#     masks = []
#     ids = []
#     for i in track(range(len(img_paths)), description="Load Data"):
#         image = Image.open(img_paths[i])
#         mask = Image.open(mask_paths[i])
#         image = np.array(image)
#         image = image.astype(np.uint8)
#         mask = np.array(mask, dtype=np.int64)
#         images.append(image)
#         masks.append(mask)
#         ids.append(i)
#     return images, masks, ids
#
# def cellpose(root_path, train_pkl_path, test_pkl_path, image_size):
#     root_path = Path("/data/zhoupeilin/data/CellPose/")
#     train_data_path = root_path / "train"
#     test_data_path = root_path / "test"
#
#     train_images, train_masks, train_ids = get_data(train_data_path)
#     test_images, test_masks, test_ids = get_data(test_data_path)
#     # train_images, train_masks = noramlize_resize_flow(train_images, train_masks, image_size)
#     # test_images, test_masks = noramlize_resize_flow(test_images, test_masks, image_size)
#
#     with open(train_pkl_path, "wb") as f:
#         pickle.dump((train_images, train_masks), f)
#     with open(test_pkl_path, "wb") as f:
#         pickle.dump((test_images, test_masks), f)

# if __name__ == "__main__":
#     cellpose_folder_path = Path(r"/data/zhoupeilin/data/CellPose/")
#     cellpose_train_path = cellpose_folder_path / "cellpose_train.pkl"
#     cellpose_test_path = cellpose_folder_path / "cellpose_test.pkl"
#     cellpose(cellpose_folder_path, cellpose_train_path, cellpose_test_path, 512)