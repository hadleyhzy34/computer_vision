import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
from torch.autograd import Variable
from torch import optim
from PIL import Image
import cv2 as cv

#resnet18 for residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        # print('basicblock is called',in_planes,planes)
        super(BasicBlock, self).__init__()
        #adding padding one to keep input output same dimension
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.in_planes = in_planes
        self.planes = planes

    def forward(self, x):
        residual = x
        # print('residual size is:',residual.size())
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print('out size is: ',out.size())
        print("input planes are: ",self.in_planes)
        print("output planes are: ",self.planes)
        #add residual only for the same input/output channels
        if self.in_planes == self.planes:
            out += residual
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        '''instance of block is initialized here, here input argument is class, not object'''
        # print('resnet18 is called',block.__name__,num_blocks)
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print('input planes number is: ',self.in_planes)
        out = self.layer1(out)
        # print('second input plane number is: ',self.in_planes)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train(dataloader,testloader,epochs,model):
    device = torch.device("cpu")
    net = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    test_images,test_labels = testloader
    #training process
    for epoch in range(epochs):
        running_loss = 0.0
        for step,data in enumerate(dataloader,start=0):
            print(f'current step is: {step}')
            #input data
            inputs,labels=data
            #zero grad
            optimizer.zero_grad()
            #forward
            output = net(inputs)
            #calculate loss
            loss = loss_function(output,labels)
            loss.backward()
            #update weight
            optimizer.step()

            #print current status
            running_loss += loss.item()
            if step % 100 == 99:
                with torch.no_grad():
                    res = net(test_images)
                    predict_y = torch.max(res,dim=1)[1]
                    accuracy = (predict_y == test_labels).sum().item() / test_labels.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f'%(epoch + 1, step + 1, running_loss / 100, accuracy))
                    running_loss = 0.0


    print('-------------------------------------------------------------')
    print('Finished Training')
    torch.save(net.state_dict(),'ResNet18.pth')


def dataloader():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    #download and load the training data
    trainset = datasets.FashionMNIST('..\\data\\',download=True,train=True,transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)

    testset = datasets.FashionMNIST('..\\data\\',train=False,transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = 10000, shuffle=True)
    #randomly select images for test
    iter_test = iter(test_loader)
    test_data = iter_test.next()
    return train_loader,test_data

def img_processing(img,model):
    #print("image shape is: ", img.shape)
    img = cv.resize(img,(28,28),interpolation = cv.INTER_AREA)
    img = np.array(img, dtype=np.float32)
    #print(img[0])
    #!it's super important to flip greyscale color since trained data is black background and white foreground
    img = 255 - img
    #print(img[0])
    cv.imwrite('.\\data\\test_proc.jpg',img)
    if model == 'torch':
        return img

def run():
      img = np.array(Image.open('.\\data\\test1.jpg').convert('L'),'f')

      #image processing for test image
      testimg = img_processing(img,'torch')
      #print("test image size is: ", testimg.shape)
      model = ResNet18(BasicBlock,[2,2,2,2])
      model.load_state_dict(torch.load('ResNet18.pth'))
      #evaluate mode
      model.eval()
      transform = transforms.ToTensor()
      testimg = Variable(transform(testimg))
      testimg = testimg.reshape([1,1,28,28])
      res = model(testimg)
      _,predicted = torch.max(res,1)
      target_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot',}
      msg = f'this clothes should be: {target_dict[int(predicted)]}'
      print(msg)
      return msg

model = ResNet18(BasicBlock,[2,2,2,2])
data_loader,test_loader = dataloader()
train(data_loader, test_loader, 1, model)
# run()

