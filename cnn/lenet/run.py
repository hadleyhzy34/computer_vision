#test
      model = LeNet_5()
      data_loader,test_loader = dataloader()
      train(data_loader, test_loader, epoch, model)


      #read the image
      try:
      ¦   img = np.array(Image.open(request.files['file'].stream).convert('L'),'f')
--    except Exception:
      ¦   return {'code': 10000, 'msg':'image is loaded not successfully, please upload it again'}

      #image processing for test image
      testimg = img_processing(img,'torch')
      #print("test image size is: ", testimg.shape)
      model = LeNet_5()
      model.load_state_dict(torch.load('app/util/tf_fashion_mnist/checkpoints/Lenet_5.pth'))
      #evaluate mode
      model.eval()
      transform = transforms.ToTensor()
      testimg = Variable(transform(testimg))
      testimg = testimg.reshape([1,1,28,28])
      res = model(testimg)
      _,predicted = torch.max(res,1)
--    target_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot',}
      msg = f'this clothes should be: {target_dict[int(predicted)]}'
      return {'code':0,'msg':msg}

  #load the model
--    try:
      ¦   model = tf.keras.models.load_model('app/util/tf_fashion_mnist/checkpoints/model.h5')
--    except Exception:
      ¦   return {'code':10000, 'msg':'model not found, please train the model first'}

      #model prediction
      predictions = model.predict(testimg)

      # set the target
--    target_dict = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot',}
      msg = f'this clothes should be: {target_dict[np.argmax(predictions[0])]}'

      return {'code':0,'msg':msg}
