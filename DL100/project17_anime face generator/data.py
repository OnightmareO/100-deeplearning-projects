import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
from matplotlib import pyplot as plt

class AnimefaceDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for img_name in os.listdir(root_dir):
            img_path = os.path.join(root_dir, img_name)
            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
if __name__ == '__main__':
    data = AnimefaceDataset('./data/anime face')
    print(min(face_data.size[0] for face_data in data))
    print(min(face_data.size[1] for face_data in data))
    sel_idx = 10
    img = data[sel_idx]
    # Print its shape
    print(f'Image shape: {img.size}') 

    # 显示多张图像
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(data))):
        img = data[i]
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
