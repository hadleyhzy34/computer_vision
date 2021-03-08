>>import tensorflow as tf
>>import numpy as np
>>from torchvision import datasets, transforms
>>import torch

  # load the data
--def load_mnist():
--    ((train_data, train_labels),(test_data,test_labels)) = tf.keras.datasets.fashion_mnist.load_data()

      #prepare the data
      train_data = train_data.reshape((-1,28,28,1))
      test_data = test_data.reshape((-1,28,28,1))

      train_data = train_data/np.float32(255)
      train_labels = train_labels.astype(np.int32)
      test_data = test_data/np.float32(255)
      test_labels = test_labels.astype(np.int32)
      return train_data,train_labels,test_data,test_labels

--def dataloader():
      transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

      #download and load the training data
--    trainset = datasets.FashionMNIST('/home/user/git/cip/data/vision/fashion',download=True,train=True,transform=transform)
      train_loader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle=True)
--
--    testset = datasets.FashionMNIST('/home/user/git/cip/data/vision/fashion',train=False,transform=transform)
      test_loader = torch.utils.data.DataLoader(testset, batch_size = 10000, shuffle=True)
      #randomly select images for test
      iter_test = iter(test_loader)
      test_data = iter_test.next()
      return train_loader,test_data
