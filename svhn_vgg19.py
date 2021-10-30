from torchvision import transforms, datasets
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim

# SVHN VGG-19


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def SVHNDataLoader():
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4309, 0.4302, 0.4463), (0.1965, 0.1983, 0.1994))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524, 0.4525, 0.4690), (0.2194, 0.2266, 0.2285))
        ])
    }

    if device == 'cpu':
        trainset = torchvision.datasets.SVHN(root='/Users/changzhang/PycharmProjects/bupt/DATA/SVHN/', split='train',
                                            download=False, transform=data_transforms['train'])
        testset = torchvision.datasets.SVHN(root='/Users/changzhang/PycharmProjects/bupt/DATA/SVHN/', split='test',
                                               download=False, transform=data_transforms['val'])
    else:
        trainset = torchvision.datasets.SVHN(root='/home/zc/changzhang/bupt/DATA/SVHN/', split='train',
                                                download=False, transform=data_transforms['train'])
        testset = torchvision.datasets.SVHN(root='/home/zc/changzhang/bupt/DATA/SVHN/', split='test',
                                               download=False, transform=data_transforms['val'])
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    return trainloader, testloader

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


def train_model():
    trainloader, testset_loader = SVHNDataLoader()

    net = VGG('VGG16')
    # print('# net parameters:', sum(param.numel() for param in net.parameters()))
    #
    # net2 = MobileNetV3(mode='large', classes_num=10, input_size=32,
    #                   width_multiplier=1, dropout=0.2)
    # print('# net2 parameters:', sum(param.numel() for param in net2.parameters()))

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.35, momentum=0.9, weight_decay=6e-5)

    EPOCH = 20
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

            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (
            epoch, step, sum_loss / (step + 1), 100. * accuracy / total))
        print("___________________________________________________")


if __name__ == '__main__':
    train_model()

