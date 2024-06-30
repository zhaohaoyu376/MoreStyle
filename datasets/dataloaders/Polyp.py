import os
import os.path as osp
from datasets.utils.transform_multi import *
from torch.utils.data import Dataset
from torchvision import transforms
from batchgenerators.utilities.file_and_folder_operations import *

class PolypDataset(Dataset):
    def __init__(self, root, source_csv, mode='train',img_list=None, label_list=None, transform=None,patch_size=(224,224)):
        super(PolypDataset, self).__init__()
        # data_path = osp.join(root, data_dir)
        self.root = root
        self.img_list = img_list
        self.label_list = label_list
        self.len = len(img_list)
        self.target_size = patch_size
            
        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize(patch_size),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((224, 224)),
                   ToTensor(),
               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize(patch_size),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):

        img_file = join(self.root, self.img_list[index])
        label_file = join(self.root, self.label_list[index])

        img = Image.open(img_file).convert('RGB')
        label = Image.open(label_file).convert('L')
        data = {'image': img, 'label': label}

        if self.transform:
            data = self.transform(data)

        data['name'] = img_file

        return data


    def __len__(self):
        return len(self.img_list)