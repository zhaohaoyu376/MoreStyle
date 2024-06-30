from torch.utils import data
import numpy as np
from PIL import Image
from batchgenerators.utilities.file_and_folder_operations import *

class MM_set(data.Dataset):
    def __init__(self, root, source_csv, target_size=(512, 512), img_normalize=True):
        super().__init__()
        self.root = root
        self.root = root
        name = source_csv[0].split('.')[0]+'.pkl'
        with open(name, 'rb') as f:
            data = pickle.load(f)
            self.img_list, self.label_list, self.name = data
        self.len = len(self.img_list)
        # self.img_list = img_list
        # self.label_list = label_list
        # self.len = len(img_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        # img_file = join(self.root, self.img_list[item])
        # label_file = join(self.root, self.label_list[item])
        img_npy = self.img_list[item]
        label_npy = self.label_list[item]
        img_file = self.name[item]
        if self.img_normalize:
            for i in range(img_npy.shape[0]):
                img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()

        mask = np.zeros_like(label_npy)
        mask[label_npy == 85] = 1
        mask[label_npy == 170] = 2
        mask[label_npy == 255] = 3

        img_npy = np.expand_dims(img_npy, axis=0)

        # with Image.open(img_file) as img, Image.open(label_file) as label:
        #     # img = Image.open(img_file)
        #     # label = Image.open(label_file)
        #     img = img.resize(self.target_size)
        #     label = label.resize(self.target_size, resample=Image.NEAREST)
        #     img_npy = np.array(img).astype(np.float32)
        #     # img_npy = np.array(img).astype(np.float32)
        #     # if self.img_normalize:
        #     #     for i in range(img_npy.shape[0]):
        #     #         img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        #
        #     if self.img_normalize:
        #         try:
        #             for i in range(img_npy.shape[0]):
        #                 img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()
        #         except:
        #             print(img_file)
        #     label_npy = np.array(label)
        #     mask = np.zeros_like(label_npy)
        #     mask[label_npy == 85] = 1
        #     mask[label_npy == 170] = 2
        #     mask[label_npy == 255] = 3
        #
        #     img_npy = np.expand_dims(img_npy,axis=0)

            # img.close()
            # label.close()

        return img_npy, mask[np.newaxis], img_file