import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import SimpleFashionMnist

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3529,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleFashionMnist().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), './models/3_simple_fashionmnist.pth')
print("Model saved to ./models/3_simple_fashionmnist.pth")

model.load_state_dict(torch.load('./models/3_simple_fashionmnist.pth'))
print("Model loaded from ./models/3_simple_fashionmnist.pth")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')