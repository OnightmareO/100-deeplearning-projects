import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt

class AnimalDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform =transform
        self.image_paths = []
        self.labels = []
        self.class_name = []
        for label, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir,class_dir)
            self.class_name.append(class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path,img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.labels)

    def get_label_description(self,label:int):
        return self.class_name[label]        
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,label
    
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
    data = AnimalDataset('./data/animal_data')
    print(len(data))
    print(min(animal_data[0].size[0] for animal_data in data))
    print(min(animal_data[0].size[1] for animal_data in data))
    sel_idx = 0
    img, label = data[sel_idx]
    print(f'Label: {label}')
    print(f'Classname: {data.get_label_description(label)}')
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


