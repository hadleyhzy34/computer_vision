#tutorial on understanding vgg
#https://zhuanlan.zhihu.com/p/41423739
# https://www.jb51.net/article/168529.htm
# https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html
# https://blog.csdn.net/Remoa_Dengqinyi/article/details/109558378?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
# https://github.com/lavendelion/vgg16_for_CIFAR10_with_pytorch/blob/master/vgg16.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable

class vgg(nn.Module):  
    """
    vgg model
    """
    def __init__(self,layers,num_classes=1000):
        super(vgg,self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64,layers[0])
        self.conv3_128 = self.__make_layer(128,layers[1])
        self.conv3_256 = self.__make_layer(256,layers[2])
        self.conv3_512a = self.__make_layer(512,layers[3])
        self.conv3_512b = self.__make_layer(512,layers[4])
        self.fc1 = nn.Linear(512,4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,num_classes)
    
    def __make_layer(self,channels,num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels,channels,3,stride=1,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        # print(x.size())
        out = self.conv3_64(x)
        out = F.max_pool2d(out,2)
        # print(out.size())
        out = self.conv3_128(out)
        out = F.max_pool2d(out,2)
        # print(out.size())
        out = self.conv3_256(out)
        out = F.max_pool2d(out,2)
        # print(out.size())
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        # print(out.size())
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        # print(out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
        # return F.softmax(self.fc3(out))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
def vgg_11():
    return vgg([1,1,2,2,2], num_classes=10)
    
def test():
    net = vgg_11()
    summary(net, (3,224,224))

 
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)

# test()

#data processing
print('==> Preparing data..')
# #image transform
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
# print('train data is set')

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
# print(testloader)
# print('test data is set')

# 当时LR=0.01遇到问题，loss逐渐下降，但是accuracy并没有提高，而是一直在10%左右，修改LR=0.00005后，该情况明显有所好转，准确率最终提高到了
# 当LR=0.0005时，发现准确率会慢慢提高，但是提高的速度很慢，这时需要增加BATCH_SIZE，可以加快训练的速度，但是要注意，BATCH_SIZE增大会影响最终训练的准确率，太大了还可能也会出现不收敛的问题
# 另外，注意每次进入下一个EPOCH都会让准确率有较大的提高，所以EPOCH数也非常重要，需要让网络对原有数据进行反复学习，强化记忆
#
# 目前，调试的最好的参数是BATCH_SIZE = 500  LR = 0.0005  EPOCH = 10  最终准确率为：69.8%    用时：

def getData():  # 定义数据预处理
    # transforms.Compose([...])就是将列表[]里的所有操作组合起来，返回所有操作组合的句柄
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32),  # 将图像随机裁剪为不同大小(默认0.08~1.0和宽高比(默认3/4~4/3)，224是期望输出的图像大小
        transforms.RandomHorizontalFlip(),  # 以一定几率(默认为0.5)水平翻转图像
        transforms.ToTensor(),  # 将图像数据或数组数据转换为tensor数据类型
        transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 标准化tensor数据类型的图像数据，其中mean[i],std[i]分别表示图像第i个通道的均值和标准差，
                             std=[1, 1, 1])])  #标准化公式为input[channel] =(input[channel] - mean[channel])/std[channel]
    trainset = torchvision.datasets.CIFAR10(root='F:\\developments\\computer_vision\\data\\', train=True, transform=transform, download=True)  # 获取CIFAR10的训练数据
    testset = torchvision.datasets.CIFAR10(root='F:\\developments\\computer_vision\\data\\', train=False, transform=transform, download=True)  # 获取CIFAR10的测试数据

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True)  # 将数据集导入到pytorch.DataLoader中
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)  # 将测试集导入到pytorch.DataLoader中
    return train_loader, test_loader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('classes is printed')

# training process
def train(epochs):
    train_loader,test_loader = getData()
    print('\nEpoch: %d' % epochs)
    # switch to train mode
    net = vgg_11()
    net.train()
    print(net)
    print('vgg model is built')

    # print(testloader[0].size())
    # _,(test_images,test_labels) = test_loader
    # print(test_images.size())
    #optimizer,criterion 
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    
    #train mode
    # net.train()
    # training
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        accuracy = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # print('if its ever been started')
            # set optimizer gradient to be zero
            optimizer.zero_grad()
            #set up compute graph
            inputs,targets = Variable(inputs),Variable(targets)
            #forward
            outputs = net(inputs)
            #calculate loss
            loss = criterion(outputs,targets)
            #backpropagation
            loss.backward()
            #update param
            optimizer.step()
            #loss statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # predicted 当前图像预测的类别
            total += targets.size(0)
            accuracy += (predicted == targets).sum()
            # tensor数据(在GPU上计算的)如果需要进行常规计算，必须要加.cpu().numpy()转换为numpy类型，否则数据类型无法自动转换为float
            print("epoch %d | step %d: loss = %.4f, the accuracy now is %.3f %%." % (epoch, batch_idx, running_loss/(batch_idx+1), 100.*accuracy.cpu().numpy()/total))
            # # 数据统计
            # _, top_k = torch.topk(outputs,k=3,dim=1)
            # predicted = top_k[:,0]
            # # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()

            # print('[%d, %5d] train_loss: %.3f  train_accuracy: %.3f'%(epoch + 1, batch_idx + 1, running_loss / (batch_idx + 1), 100.*correct/total))

        test_loss = 0
        correct = 0
        total = 0
        #evaluate
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            #set up compute graph
            inputs,targets = Variable(inputs),Variable(targets)
            #forward
            outputs = net(inputs)
            #calculate loss
            loss = criterion(outputs,targets)
            #loss statistics
            test_loss += loss.item()

                
            # 数据统计
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f'%(epoch + 1, batch_idx + 1, test_loss / (batch_idx + 1), 100.*correct/total))

# train(1,trainloader,testloader)
train(10)

