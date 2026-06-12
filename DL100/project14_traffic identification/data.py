import torch
from torch.utils.data import Dataset,Subset
import os 
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
class TrafficSignDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        csv_path = os.path.join(self.root_dir, 'annotations.csv')
        df = pd.read_csv(csv_path)
        # 一个字典，key为file_name，value为category
        label_map = dict(zip(df['file_name'], df['category']))

        self.image_folder = os.path.join(self.root_dir, 'images')
        for image_name in os.listdir(self.image_folder):
            self.image_path = os.path.join(self.image_folder, image_name)
            self.image_paths.append(self.image_path)
            self.labels.append(label_map.get(image_name, -1))

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
    data = TrafficSignDataset('./data/14_traffic_sign')
    print(len(data))
    print(min(traffic_data[0].size[0] for traffic_data in data))
    print(min(traffic_data[0].size[1] for traffic_data in data))
    sel_idx = 0
    img, label = data[sel_idx]
    print(f'Label: {label}')
    # Print its shape
    print(f'Image shape: {img.size}')  # PIL image size is (width, height)

    # 显示多张图像
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(data))):
        img, label = data[i]
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

