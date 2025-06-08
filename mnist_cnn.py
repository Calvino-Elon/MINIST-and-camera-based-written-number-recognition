import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import numpy as np

# 固定所有随机种子（让每次结果一样）
torch.manual_seed(42)     # PyTorch
random.seed(42)           # Python随机
np.random.seed(42)        # NumPy


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

# 修正后的模型
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入: [1, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输入: [32, 28, 28]
        self.pool = nn.MaxPool2d(2, 2)  # 输出尺寸减半
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 输入维度必须是 64*7*7
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # [batch, 32, 28, 28]
        x = self.pool(x)              # [batch, 32, 14, 14]
        x = self.relu(self.conv2(x))  # [batch, 64, 14, 14]
        x = self.pool(x)              # [batch, 64, 7, 7]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)     # 展平为 [batch, 64*7*7=3136]
        x = self.relu(self.fc1(x))    # [batch, 128]
        x = self.dropout(x)
        x = self.fc2(x)               # [batch, 10]
        return x

# 训练和测试
model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 初始化最佳准确率为0
best_accuracy = 0.0

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy  # 返回准确率用于比较

# 训练5个epoch
for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, criterion, epoch)
    current_accuracy = test(model, device, test_loader)
    
    # 如果当前准确率高于最佳准确率，则保存模型
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'New best model saved with accuracy: {best_accuracy:.2f}%')

print(f'Training finished. Best accuracy: {best_accuracy:.2f}%')