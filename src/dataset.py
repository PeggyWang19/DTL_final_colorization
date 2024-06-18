from torch.utils.data import Dataset
from PIL import Image

import os

def list_files(directory):
    files = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path):
            files.append(entry)
    return files


class BlackColorImages(Dataset):
    
    def __init__(self, root, transform=None):
        
        b_img_folder = os.path.join(root, 'images_black')
        c_img_folder = os.path.join(root, 'images_color')
            
        # 遍历黑白图片的文件名
        file_names = list_files(b_img_folder)
                
        self.b_imgs_path = []
        self.c_imgs_path = []
        
        for name in file_names:
            self.b_imgs_path.append(os.path.join(b_img_folder, name))
            self.c_imgs_path.append(os.path.join(c_img_folder, name))
            
        self.transform = transform

    def __len__(self):
        return len(self.b_imgs_path)

    def __getitem__(self, idx):
        
        b_img_path = self.b_imgs_path[idx]
        c_img_path = self.c_imgs_path[idx]
        
        b_img = Image.open(b_img_path)
        c_img = Image.open(c_img_path)
        
        if self.transform:
            b_img = self.transform(b_img)
            c_img = self.transform(c_img)

        return b_img, c_img