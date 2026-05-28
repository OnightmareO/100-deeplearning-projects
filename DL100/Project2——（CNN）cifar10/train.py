import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2023, 0.1994, 0.2010])
])
# train_dataset 是一个 torchvision.datasets.CIFAR10 对象
#它内部主要包含：train_dataset.data 和 train_dataset.targets 以及 train_dataset.classes
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

'''
plt.figure(figsize=(20,10))
for i in range(20):
    plt.subplot(5,10,i+1)
    plt.xticks([]) #关闭 x 轴刻度标签
    plt.yticks([]) #关闭 y 轴刻度标签
    plt.grid(False) #关闭网格线
    plt.imshow(train_dataset.data[i], cmap=plt.cm.binary) # 显示图像，cmap=plt.cm.binary 是为了显示灰度图像，如果是彩色图像可以省略
    plt.xlabel(train_dataset.classes[train_dataset.targets[i]]) # 显示标签
plt.show()
'''

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

image0, label = train_dataset[0] # Get the first image
print(f"train_dataset shape:        {train_dataset.data.shape}") # (50000, 32, 32, 3)
print(f"test_dataset shape:         {test_dataset.data.shape}") # (10000, 32, 32, 3)
print(f"Image type:        {type(image0)}")
# Since `image0` is a PIL Image object, its dimensions are accessed using the .size attribute.
print(f"Image Dimensions:  {image0.size}")
print(f"Label Type:        {type(label)}")
print(f"Label value:       {label}")

from model import SimpleCifar
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
model = SimpleCifar().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

'''
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.shape 是 [64, 3, 32, 32], target.shape 是 [64]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #torch.max(output.data, 1) 返回 output.data 中每行的最大值和对应的索引
        _, predicted = torch.max(output.data, 1)
        # batch_size=target.size(0)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), './models/2_simple_cifar.pth')
print("Model saved to ./models/2_simple_cifar.pth")
'''

model.load_state_dict(torch.load('./models/2_simple_cifar.pth'))
print("Model loaded from ./models/2_simple_cifar.pth")
model.eval()
correct = 0 
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        #torch.max(output.data, 1) 返回 output.data 中每行的最大值和对应的索引
        _, predicted = torch.max(output.data, 1)
        # batch_size=target.size(0)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')