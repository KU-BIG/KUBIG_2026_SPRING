#속도가 훨씬 빠름...

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # layer1: 64 -> 64 (stride 1)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1)
        )
        
        # layer2: 64 -> 128 (stride 2 -> 크기 줄어듦)
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        
        # layer3: 128 -> 256 (stride 2) 
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )
        
        # layer4: 256 -> 512 (stride 2)
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )
        
        # 마지막 분류기
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
       
        out = F.adaptive_avg_pool2d(out, (1, 1)) #입력이 몇이든 상관없으니 무조건 1x1로
        out = out.view(out.size(0), -1) # 펴주기
        out = self.fc(out)
        return out

if __name__ == '__main__':
    # 1. 설정
    device = torch.device("cpu")
    print(f"Device: {device}")
    
    batch_size = 64
    learning_rate = 0.001
    max_epochs = 5

    # 2. 데이터 준비 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 데이터셋 다운로드
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    full_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # 데이터 1/10로 줄이기 (속도 향상용)
    train_indices = list(range(0, len(full_trainset), 10))
    test_indices = list(range(0, len(full_testset), 10))
    
    trainset = torch.utils.data.Subset(full_trainset, train_indices)
    testset = torch.utils.data.Subset(full_testset, test_indices)

    # 로더 생성
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)

    print(f"학습 데이터: {len(trainset)}개 (Subset), 테스트 데이터: {len(testset)}개 (Subset)")

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining Started...")
    
    for epoch in range(max_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()       # 기울기 초기화
            outputs = model(inputs)     # 예측
            loss = criterion(outputs, labels) # 오차 계산
            loss.backward()             # 역전파
            optimizer.step()            # 가중치 갱신

            # 통계 계산
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 20 == 19:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 20:.3f}')
                running_loss = 0.0
        
        epoch_acc = 100 * correct / total
        end_time = time.time()
        print(f"Epoch {epoch + 1} Finished ({end_time - start_time:.1f}sec) | Accuracy: {epoch_acc:.2f}%")

    print('Finished Training')