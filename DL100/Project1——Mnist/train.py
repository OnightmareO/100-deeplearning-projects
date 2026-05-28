import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# 1. 定义数据预处理
transform= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

image0, label = train_dataset[0] # Get the first image
print(f"train_dataset shape:        {train_dataset.data.shape}")
print(f"test_dataset shape:         {test_dataset.data.shape}")
print(f"Image type:        {type(image0)}")
# Since `image0` is a PIL Image object, its dimensions are accessed using the .size attribute.
print(f"Image Dimensions:  {image0.size}")
print(f"Label Type:        {type(label)}")
print(f"Label value:       {label}")

# 2. 定义模型
from model import SimpleMNIST
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMNIST().to(device)
'''
# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # target 是对应的数字标签,data 是对应的图像数据
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# 5. 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%')

# 6. 保存模型
torch.save(model.state_dict(), './models/1_simple_mnist.pth')
print("Model saved to ./models/1_simple_mnist.pth")
'''
# 7. 加载模型
model.load_state_dict(torch.load('./models/1_simple_mnist.pth'))
print("Model loaded from ./models/1_simple_mnist.pth")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        #Data shape: torch.Size([1000, 1, 28, 28]), Target shape: torch.Size([1000])
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f'Accuracy: {100 * correct / total:.2f}%')

