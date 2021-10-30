import time
import torch
import torchvision
import torch.utils.data as Data
from torch import nn, optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    # 定义前向传播的过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化
                nn.init.constant_(m.bias, 0)  # 初始化偏重为0


def getData():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    if device == 'cuda':
        trainset = torchvision.datasets.FashionMNIST(root='/home/zc/changzhang/bupt/DATA/', train=True,
                                                     download=False, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='/home/zc/changzhang/bupt/DATA/', train=False,
                                                    download=False, transform=transform_test)
    else:
        trainset = torchvision.datasets.FashionMNIST(root='/Users/changzhang/PycharmProjects/bupt/DATA/', train=True,
                                                download=False, transform=transform_train)
        testset = torchvision.datasets.FashionMNIST(root='/Users/changzhang/PycharmProjects/bupt/DATA/', train=False,
                                               download=False, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    # 不再求梯度
    with torch.no_grad():
        for X, y in data_iter:
            # print(X.shape)
            # print(y.shape)
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n


def test(net, testdata):
    correct, total = .0, .0
    for inputs_cpu, labels_cpu in testdata:
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.cuda()
        net.eval()
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return float(correct) / total

def train():
    net = AlexNet()
    print('==> Building model..')
    print('# net parameters:', sum(param.numel() for param in net.parameters()))

    # print(net)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    EPOCH = 10
    # Train the model
    for epoch in range(EPOCH):
        sum_loss = 0.
        total = 0.
        accuracy = 0.
        for step, (inputs, labels) in enumerate(train_loader):
            if device == 'cuda':
                inputs = inputs.cuda()
                labels = labels.cuda()

            output = net(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            sum_loss += loss.item()
            total += labels.size(0)
            accuracy += (predicted == labels).sum()
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, step, sum_loss/(step+1), 100.*accuracy/total))

        acc = test(net, test_loader)
        print("")
        print("___________________________________________________")
        print("epoch %d : test accuracy = %.4f %%" % (epoch, 100 * acc))
        print("---------------------------------------------------")
    print('Finished Training')



if __name__ == '__main__':
    train_loader, test_loader = getData()
    train()
