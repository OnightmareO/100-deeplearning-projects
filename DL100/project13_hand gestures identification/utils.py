from collections import defaultdict
import torch
from data import HandGestureDataset,SubsetDataset
from torch.utils.data import Dataset,DataLoader,Subset
import torchvision.transforms as transforms

def get_mean_std(dataset:Dataset):

    preprocess = transforms.Compose(
        [transforms.Resize((256,256)),
         transforms.CenterCrop((224,224)),
         transforms.ToTensor()
         ]
    )

    means = []
    stds = []
    for img, _ in dataset:
        img = preprocess(img)
        means.append(img.mean(dim=(1,2)))
        stds.append(img.std(dim=(1,2)))
    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).std(dim=0)

    return mean.tolist(),std.tolist()

def get_transformations(mean,std):

    main_tfs = [
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor()    
    ]

    augmentation_tfs = [
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ] 

    main_tranformation = transforms.Compose(main_tfs)
    transform_with_augmentation = transforms.Compose(augmentation_tfs)

    return main_tranformation, transform_with_augmentation

g = torch.Generator()
g.manual_seed(42)
def get_dataloader(dataset,batch_size,val_part,test_part,main_transformation,aug_transformation):

    total_size = len(dataset)
    val_size = total_size * val_part
    test_size = total_size * test_part
    train_size = total_size - val_size - test_size

    train_dataset,val_dataset,test_dataset = stratified_split(dataset, val_part, test_part, generator=g)
    # 为训练集应用数据增强转换。
    train_dataset = SubsetDataset(train_dataset,transform=aug_transformation)
    val_dataset = SubsetDataset(val_dataset,transform=main_transformation)
    test_dataset = SubsetDataset(test_dataset,transform=main_transformation)

    train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size,shuffle=True)

    return train_loader,val_loader,test_loader


def stratified_split(dataset, val_part, test_part, generator=None):
    label_indices = defaultdict(list)
    for idx,label in enumerate(dataset.labels):
        label_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for label,indices in label_indices.items():
        perm = torch.randperm(len(indices),generator=generator).tolist()
        indices = [indices[i] for i in perm]
        
        total = len(indices)
        val_size = int(total * val_part)
        test_size = int(total * test_part)
        train_size = total - val_size - test_size

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
    
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset,test_indices)

    return train_ds,val_ds,test_ds
