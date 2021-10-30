import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 VGG-16

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# vgg16的网络结构参数，数字代表该层的卷积核个数，'M'代表该层为最大池化层
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def getData():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if device == 'cpu':
        trainset = torchvision.datasets.CIFAR10(root='/Users/changzhang/PycharmProjects/bupt/DATA/CIFAR-10/', train=True,
                                            download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/Users/changzhang/PycharmProjects/bupt/DATA/CIFAR-10/',
                                               train=False,
                                               download=False, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(root='/home/zc/changzhang/bupt/DATA/CIFAR-10/', train=True,
                                                download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/home/zc/changzhang/bupt/DATA/CIFAR-10/', train=False,
                                               download=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


# Training
def train():
    trainloader, testset_loader = getData()

    print('==> Building model..')
    net = VGG('VGG16')
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    EPOCH = 30
    # Train the model
    for epoch in range(EPOCH):
        sum_loss = 0.
        total = 0.
        accuracy = 0.
        for step, (inputs, labels) in enumerate(trainloader):
            if device != 'cpu':
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
            # print((predicted == labels))
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, step, sum_loss/(step+1), 100.*accuracy/total))
        print("___________________________________________________")
    #     acc = test(net, testset_loader)
    #     print("")
    #     print("___________________________________________________")
    #     print("epoch %d : test accuracy = %.4f %%" % (epoch, 100 * acc))
    #     print("---------------------------------------------------")
    # print('Finished Training')


def test(net, testdata):
    """检测当前网络的效果"""
    correct, total = .0, .0
    for inputs_cpu, labels_cpu in testdata:
        inputs = inputs_cpu.cuda()
        labels = labels_cpu.cuda()
        net.eval()  # 有些模块在training和test/evaluation的时候作用不一样，比如dropout等等。
                    # net.eval()就是将网络里这些模块转换为evaluation的模式
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        #print(predicted)
    # net.train()
    # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
    # return float(correct.cpu().numpy()) / total
    return float(correct) / total

if __name__ == '__main__':
    train()
