>>import torch
>>import torchvision
>>import torch.nn as nn
>>from app.util.tf_fashion_mnist.model import LeNet_5
>>import torch.optim as optim
>>import torchvision.transforms as transforms
>>from torch import optim

--def train(dataloader,testloader,epochs,model):
      device = torch.device("cpu")
      net = model.to(device)
      loss_function = nn.CrossEntropyLoss()
      optimizer = optim.Adam(net.parameters(),lr=0.001)
      test_images,test_labels = testloader
      #training process
      for epoch in range(epochs):
      ¦   running_loss = 0.0
      ¦   for step,data in enumerate(dataloader,start=0):
      ¦   ¦   #input data
      ¦   ¦   inputs,labels=data
      ¦   ¦   #zero grad
      ¦   ¦   optimizer.zero_grad()
      ¦   ¦   #forward
      ¦   ¦   output = net(inputs)
      ¦   ¦   #calculate loss
      ¦   ¦   loss = loss_function(output,labels)
      ¦   ¦   loss.backward()
      ¦   ¦   #update weight
      ¦   ¦   optimizer.step()

      ¦   ¦   #print current status
      ¦   ¦   running_loss += loss.item()
      ¦   ¦   if step % 100 == 99:
      ¦   ¦   ¦   with torch.no_grad():
      ¦   ¦   ¦   ¦   res = net(test_images)
      ¦   ¦   ¦   ¦   predict_y = torch.max(res,dim=1)[1]
      ¦   ¦   ¦   ¦   accuracy = (predict_y == test_labels).sum().item() / test_labels.size(0)
--    ¦   ¦   ¦   ¦   print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f'%(epoch + 1, step + 1, running_loss / 100, accuracy))
      ¦   ¦   ¦   ¦   running_loss = 0.0


      print('-------------------------------------------------------------')
      print('Finished Training')
      torch.save(net.state_dict(),
      ¦   ¦   ¦  'app/util/tf_fashion_mnist/checkpoints/Lenet_5.pth')
