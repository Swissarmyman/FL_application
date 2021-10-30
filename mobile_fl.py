import torch
import random
from torch import nn
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

# SVHN MobileNetV3 FL

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

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y

def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num
#
#
class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''

    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6
#
#
class H_swish(nn.Module):
    '''
    hard swish
    '''

    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

#
#
class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''

    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, BN_momentum):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'

        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if exp_size == in_channels_num:
            # Without expansion, the first depthwise convolution is omitted
            self.conv1 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0,
                          bias=False),
                # nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(
                    OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )
        else:
            # With expansion
            self.conv1 = nn.Sequential(
                # Pointwise Convolution for expansion
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=exp_size, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0,
                          bias=False),
                # nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(
                    OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )

    def forward(self, x, expand=False):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        if self.use_residual:
            out = out + x
        if expand:
            return out, out1
        else:
            return out


class MobileNetV3(nn.Module):
    '''

    '''

    def __init__(self, mode='small', classes_num=1000, input_size=224, width_multiplier=1.0, dropout=0.2,
                 BN_momentum=0.1, zero_gamma=False):
        '''
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(MobileNetV3, self).__init__()

        mode = mode.lower()
        assert mode in ['large', 'small']
        s = 2
        if input_size == 32 or input_size == 56:
            # using cifar-10, cifar-100 or Tiny-ImageNet
            s = 1

        # setting of the model
        if mode == 'large':
            # Configuration of a MobileNetV3-Large Model
            configs = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', s],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1]
            ]
        elif mode == 'small':
            # Configuration of a MobileNetV3-Small Model
            configs = [
                # kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, 'RE', s],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        first_channels_num = 16

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1280 if mode == 'large' else 1024

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier,
                                              divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum),
            H_swish()
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(
                Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL,
                           BN_momentum))
            input_channels_num = output_channels_num

        # the last stage
        last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        last_stage_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels_num, out_channels=last_stage_channels_num, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=last_stage_channels_num, momentum=BN_momentum),
            H_swish()
        )
        feature_extraction_layers.append(last_stage_layer1)

        self.featureList = nn.ModuleList(feature_extraction_layers)

        # SE Module
        # remove the last SE Module according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # feature_extraction_layers.append(SEModule(last_stage_channels_num) if mode == 'small' else nn.Sequential())

        last_stage = []
        last_stage.append(nn.AdaptiveAvgPool2d(1))
        last_stage.append(
            nn.Conv2d(in_channels=last_stage_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1,
                      padding=0, bias=False))
        last_stage.append(H_swish())

        self.last_stage_layers = nn.Sequential(*last_stage)

        ########################################################################################################################
        # Classification part

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels_num, classes_num)
        )

        '''
        self.extras = nn.ModuleList([
            InvertedResidual(576, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])
        '''

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        for i in range(9):
            x = self.featureList[i](x)
        x = self.featureList[9](x)
        for i in range(10, len(self.featureList)):
            x = self.featureList[i](x)
        x = self.last_stage_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
                if hasattr(m, 'lastBN'):
                    nn.init.constant_(m.lastBN.weight, 0.0)


class my_model():
    def __init__(self, lr, epoch_num, num_workers, train_list, test_iter):
        self.lr = lr
        self.epoch_num = epoch_num
        self.num_workers = num_workers
        self.net = MobileNetV3(mode='large', classes_num=10, input_size=32,
                          width_multiplier=1, dropout=0.2).to(device)
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
        print('Epoch %d, Client %s, Net test accuracy = %.4f %%' % (eopch, clinet, 100 * float(correct) / total))
        print('-' * 30)

    # 模型训练
    def train(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        criterion = torch.nn.CrossEntropyLoss()
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
    train_loader, testset_loader = SVHNDataLoader()
    public_net = my_model(lr=0.001, epoch_num=1, num_workers=2, train_list=[], test_iter=testset_loader)
    public_net_para = public_net.upload_para()

    client_number = 10
    select_number = 3
    communication_rounds = 20
    trainloader_list = list_segmentation(list(train_loader), client_number)
    client_data_dic = set_client_data(client_number, trainloader_list)

    for epoch in range(communication_rounds):
        # 每轮训练从客户端随机选取select_number个参与训练
        train_client_list = random.sample(list(range(client_number)), select_number)
        # print('train_client_list: %s' % train_client_list)
        net_para_list = []
        for clinet in train_client_list:
            clinet_net = my_model(lr=0.001, epoch_num=2, num_workers=2, train_list=client_data_dic[clinet], test_iter=testset_loader)
            clinet_net.download_para(public_net_para)
            clinet_net.train()
            net_para_list.append(clinet_net.upload_minus_para(public_net_para))

        public_net_para = para_average(net_para_list, public_net_para)
        public_net.download_para(public_net_para)
        public_net.test(epoch, train_client_list)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    faderated_train()


# Epoch 0, Client [4, 8, 5], Net test accuracy = 27.7313 %
# ------------------------------
# Epoch 1, Client [2, 7, 5], Net test accuracy = 81.7955 %
# ------------------------------
# Epoch 2, Client [2, 3, 6], Net test accuracy = 87.4770 %
# ------------------------------
# Epoch 3, Client [7, 0, 6], Net test accuracy = 89.0980 %
# ------------------------------
# Epoch 4, Client [2, 7, 0], Net test accuracy = 89.7165 %
# ------------------------------
# Epoch 5, Client [3, 8, 5], Net test accuracy = 90.9035 %
# ------------------------------
# Epoch 6, Client [2, 4, 1], Net test accuracy = 90.9957 %
# ------------------------------
# Epoch 7, Client [7, 9, 1], Net test accuracy = 91.5565 %
# ------------------------------
# Epoch 8, Client [1, 5, 0], Net test accuracy = 91.5104 %
# ------------------------------
# Epoch 9, Client [3, 2, 5], Net test accuracy = 92.1059 %
# ------------------------------
# Epoch 10, Client [7, 6, 3], Net test accuracy = 92.2711 %
# ------------------------------
# Epoch 11, Client [0, 5, 7], Net test accuracy = 92.2634 %
# ------------------------------
# Epoch 12, Client [6, 3, 9], Net test accuracy = 92.2903 %
# ------------------------------
# Epoch 13, Client [9, 6, 0], Net test accuracy = 92.3056 %
# ------------------------------
# Epoch 14, Client [9, 1, 6], Net test accuracy = 92.5361 %
# ------------------------------
# Epoch 15, Client [8, 2, 6], Net test accuracy = 92.9433 %
# ------------------------------
# Epoch 16, Client [0, 8, 2], Net test accuracy = 92.7167 %
# ------------------------------
# Epoch 17, Client [0, 3, 2], Net test accuracy = 92.6360 %
# ------------------------------
# Epoch 18, Client [5, 8, 4], Net test accuracy = 93.4581 %
# ------------------------------
# Epoch 19, Client [6, 9, 1], Net test accuracy = 93.2007 %


# Epoch 0, Client [4, 9, 6], Net test accuracy = 20.7360 %
# ------------------------------
# Epoch 1, Client [8, 9, 7], Net test accuracy = 81.0733 %
# ------------------------------
# Epoch 2, Client [9, 0, 6], Net test accuracy = 86.2362 %
# ------------------------------
# Epoch 3, Client [8, 4, 6], Net test accuracy = 87.7804 %
# ------------------------------
# Epoch 4, Client [4, 3, 9], Net test accuracy = 89.0865 %
# ------------------------------
# Epoch 5, Client [1, 8, 7], Net test accuracy = 89.8279 %
# ------------------------------
# Epoch 6, Client [8, 9, 3], Net test accuracy = 90.0277 %
# ------------------------------
# Epoch 7, Client [0, 9, 4], Net test accuracy = 90.9112 %
# ------------------------------
# Epoch 8, Client [6, 8, 2], Net test accuracy = 91.3530 %
# ------------------------------
# Epoch 9, Client [1, 4, 9], Net test accuracy = 91.2684 %
# ------------------------------
# Epoch 10, Client [1, 2, 5], Net test accuracy = 91.5681 %
# ------------------------------
# Epoch 11, Client [5, 4, 8], Net test accuracy = 92.1404 %
# ------------------------------
# Epoch 12, Client [6, 4, 3], Net test accuracy = 92.3594 %
# ------------------------------
# Epoch 13, Client [1, 0, 6], Net test accuracy = 92.6398 %
# ------------------------------
# Epoch 14, Client [1, 5, 3], Net test accuracy = 92.4862 %
# ------------------------------
# Epoch 15, Client [0, 7, 4], Net test accuracy = 92.7167 %
# ------------------------------
# Epoch 16, Client [0, 9, 5], Net test accuracy = 92.9164 %
# ------------------------------
# Epoch 17, Client [4, 9, 1], Net test accuracy = 92.7128 %
# ------------------------------
# Epoch 18, Client [7, 1, 6], Net test accuracy = 92.5246 %
# ------------------------------
# Epoch 19, Client [9, 7, 2], Net test accuracy = 92.6782 %
#
