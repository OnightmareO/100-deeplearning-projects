import torch
from torch.utils.data import Dataset, Subset
import os
from PIL import Image
from matplotlib import pyplot as plt
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        # 读取 self.root_dir 目录下的所有文件和子目录名
        for label,class_dir in enumerate(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir,class_dir)
            self.class_names.append(class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path,image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(label)

    def get_label_description(self,label:int):
        return self.class_names[label]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
# 方便进行不一样的tranform
class SubsetDataset(Dataset):
    def __init__(self,subset,transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    data = HandGestureDataset('./data/gestures')
    print(min(gesture_data[0].size[0] for gesture_data in data))
    print(min(gesture_data[0].size[1] for gesture_data in data))
    sel_idx = 0
    img, label = data[sel_idx]
    print(f'Label: {label}')
    print(f'Description: {data.get_label_description(sel_idx)}')
    # Print its shape
    print(f'Image shape: {img.size}')  # PIL image size is (width, height)

    # 显示多张图像
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(data))):
        img, label = data[i]
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {data.get_label_description(label)}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()