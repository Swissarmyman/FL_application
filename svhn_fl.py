import torch
import random
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# SVHN VGG-16 VGG-19

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
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
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


class my_model():
    def __init__(self, lr, epoch_num, num_workers, train_list, test_iter, model):
        self.lr = lr
        self.epoch_num = epoch_num
        self.num_workers = num_workers
        if model == 'vgg16':
            self.net = VGG('VGG16').to(device)
        else:
            self.net = VGG('VGG19').to(device)
        self.train_list = train_list
        self.test_iter = test_iter


    def download_para(self, public_net_para):
        self.net.load_state_dict(public_net_para)

    def upload_para(self):
        return self.net.state_dict()

    def upload_minus_para(self, public_net_para):
        para = self.para_minus(self.net.state_dict(), public_net_para)
        return para

    # 两个模型参数相减
    def para_minus(self, para_a, para_b):
        para = OrderedDict()
        for name in para_a:
            para[name] = para_a[name] - para_b[name]
        return para

    # 使用测试集对模型进行测试
    def test(self, eopch, clinet):
        correct, total = .0, .0
        for inputs, labels in self.test_iter:
            if device != 'cpu':
                inputs = inputs.cuda()
                labels = labels.cuda()
            self.net.eval()
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print('Communication_rounds %d, Client %s, Net test accuracy = %.4f %%' % (eopch, clinet, 100 * float(correct) / total))
        print('-' * 30)

    # 模型训练
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.35, momentum=0.9, weight_decay=6e-5)
        for epoch in range(self.epoch_num):
            for step, (inputs, labels) in enumerate(iter(self.train_list)):
                if device != 'cpu':
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                output = self.net(inputs)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

def list_segmentation(data_list, ceil_num):
    # 将训练集平均分为x份
    result = []
    length = len(data_list)
    step = int(length / ceil_num)
    for i in range(0, length, step):
        result.append(data_list[i: i + step])
    return result[:-1]


# 为每个客户端分配一份固定的数据
def set_client_data(client_number, client_data_list):
    result = {}
    for i in range(client_number):
        result[i] = client_data_list[i]
    return result


# 参数列表中所有参数的运算
def para_list_operation(tensor_list, methond='add'):
    assert len(tensor_list) >= 2, '客户端数量低于两个'
    a = tensor_list[0]
    if methond == 'add':
        for i in tensor_list[1:]:
            a += i
    return a/len(tensor_list)


def para_average(para_list, public_net_para):
    net_total_num = len(para_list)
    result = OrderedDict()
    for name in para_list[0]:
        # 2.对所有参数进行平均
        para_sum_list = [para_list[net_num][name] for net_num in range(net_total_num)]
        para_sum = para_list_operation(para_sum_list)
        result[name] = public_net_para[name] + para_sum
    return result


def faderated_train():
    model = 'vgg16'
    train_loader, test_loader = SVHNDataLoader()
    # vgg16, vgg19, mobilelarge, mobilesmall
    public_net = my_model(lr=0.001, epoch_num=1, num_workers=2, train_list=[], test_iter=test_loader, model=model)
    public_net_para = public_net.upload_para()

    client_number = 10
    select_number = 5
    communication_rounds = 10
    trainloader_list = list_segmentation(list(train_loader), client_number)
    client_data_dic = set_client_data(client_number, trainloader_list)

    for epoch in range(communication_rounds):
        # 每轮训练从客户端随机选取select_number个参与训练
        train_client_list = random.sample(list(range(client_number)), select_number)
        # print('train_client_list: %s' % train_client_list)
        net_para_list = []
        for clinet in train_client_list:
            clinet_net = my_model(lr=0.001, epoch_num=4, num_workers=2, train_list=client_data_dic[clinet], test_iter=test_loader, model=model)
            clinet_net.download_para(public_net_para)
            clinet_net.train()
            net_para_list.append(clinet_net.upload_minus_para(public_net_para))

        public_net_para = para_average(net_para_list, public_net_para)
        public_net.download_para(public_net_para)
        public_net.test(epoch, train_client_list)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    faderated_train()