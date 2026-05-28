import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader,random_split
from data import FlowerDataset, SubsetDataset

def get_mean_std(dataset:Dataset):
    '''
    Computes the mean and standard deviation of a dataset.
    Args:
        dataset (Dataset): The dataset for which to compute the mean and standard deviation.
    Returns:
        mean (list): A list of mean values for each channel.
        std (list): A list of standard deviation values for each channel.
    '''
    preprocess = transforms.Compose(
        [transforms.Resize((128, 128)), 
         transforms.ToTensor()]
    )
    means = []
    stds = []
    for img, _ in dataset:
        img = preprocess(img)
        # 遍历数据集中的每张图像，应用变换并在空间维度（高/宽）上使用 `dim=[1, 2]` 计算均值与标准差。
        means.append(img.mean(dim=(1, 2)))
        stds.append(img.std(dim=(1, 2)))
    # 将所有每图统计堆叠为张量，并对所有图像求平均以得到数据集级别的通道统计。
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)
    return mean.tolist(), std.tolist()

def get_transformations(mean, std):
    '''
    Args:
        mean (list): A list of mean values for each channel.
        std (list): A list of standard deviation values for each channel.
    Returns:
        main_transform (transforms.Compose): A composition of transformations for the main dataset.
        transform_with_augmentation (transforms.Compose): A composition of transformations that includes data augmentation.
    '''
    main_tfs = [  
        # Resize images to 128x128 pixels
        transforms.Resize((128, 128)),
        # Convert images to PyTorch tensors
        transforms.ToTensor(),
        # Normalize images using the provided mean and std
        transforms.Normalize(mean, std)
    ]  

    augmentation_tfs = [  
        # Randomly flip the image vertically
        transforms.RandomVerticalFlip(p=0.5),
        # Randomly rotate the image by ±15 degrees
        transforms.RandomRotation(degrees=15)
    ]  

    # Compose the main transformations into a single pipeline
    main_transform = transforms.Compose(main_tfs)

    transform_with_augmentation = transforms.Compose(augmentation_tfs + main_tfs)

    return main_transform, transform_with_augmentation


g = torch.Generator()
g.manual_seed(42)
def get_dataloader(dataset, batch_size,val_part,test_part,main_transform,augmentation_transform):
    '''
    Args:
        dataset (Dataset): The dataset to be loaded.
        batch_size (int): The number of samples per batch.
        val_part (float): The proportion of the dataset to be used for validation.
        test_part (float): The proportion of the dataset to be used for testing.
        main_transform: Transform to apply to validation and test splits.
        augmentation_transform: Transform to apply to the training split.

    Returns:
        dataloader (DataLoader): A DataLoader for the given dataset.
    '''
    # 计算训练、验证和测试集的大小
    total_size = len(dataset)
    val_size = int(total_size * val_part)
    test_size = int(total_size * test_part)
    train_size = total_size - val_size - test_size

    # 使用 torch.utils.data.random_split 将数据集划分为训练、验证和测试子集
    # 这种划分方法是全局随机打乱切分，而不是按类别均衡分割（分层抽样）
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_dataset = SubsetDataset(train_dataset, transform=augmentation_transform)
    val_dataset = SubsetDataset(val_dataset, transform=main_transform)
    test_dataset = SubsetDataset(test_dataset, transform=main_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader




