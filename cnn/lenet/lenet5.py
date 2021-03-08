class LeNet_5(nn.Module):
      def __init__(self):
      ¦   super().__init__()
      ¦   #default kernel is 3x3
      ¦   self.conv1 = nn.Conv2d(1,6,5)
      ¦   self.conv2 = nn.Conv2d(6,16,5)

      ¦   self.fc1 = nn.Linear(16*4*4, 120)
      ¦   self.fc2 = nn.Linear(120,84)
      ¦   self.fc3 = nn.Linear(84,10)

--    def forward(self,x):
      ¦   '''forward propogation'''
      ¦   x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
      ¦   #print('after first layer, dimension is: ', x.size())
      ¦   x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
      ¦   #print('after second layer, dimension is: ', x.size())
      ¦   x = x.view(-1, self.num_flat_features(x))
      ¦   #print('flat the layer: ', x.size())

      ¦   x = F.relu(self.fc1(x))
      ¦   x = F.relu(self.fc2(x))
      ¦   x = self.fc3(x)
      ¦   return x

--    def num_flat_features(self, x):
      ¦   '''flattern second convolution channels
      ¦   x.size() returns (batch.size,channel,image.width,image.height)
      ¦   '''
      ¦   sizes = x.size()[1:] #it returns (16,5,5)
      ¦   num_features = 1
      ¦   for size in sizes:
      ¦   ¦   num_features *= size
      ¦   return num_features
