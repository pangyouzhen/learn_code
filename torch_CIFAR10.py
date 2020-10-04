import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root="./full_data", train=True,
                                        download=True, transform=transform)
# ValueError: signal number 32 out of range  bug，不使用num_workers参数
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
testset = torchvision.datasets.CIFAR10(root="./full_data", train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
           )

import matplotlib.pyplot as plt
import numpy as np


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataiter = iter(trainloader)
images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        # [batch_size,3 RGB, 32 像素，32像素]
        # input.shape = [batch_size,3,32,32]
        labels = labels.to(device)
        # labels.shape = [batch_size]
        # 每一个batch_size 对应一类
        optimizer.zero_grad()

        outputs = net(inputs)
        # TODO
        # outputs.shape = [batch_size,class_number] labels.shape= batch_size 这里的交叉熵是怎么计算的de
        # 一维的就相当于list，二维就相当于矩阵
        # 交叉熵的公式和代码怎样对应啊
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# # imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# net = Net()
# net.load_state_dict(torch.load(PATH))
