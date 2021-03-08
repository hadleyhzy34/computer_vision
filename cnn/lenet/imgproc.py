>import numpy as np
>>import cv2 as cv

  #image processing for test image
--def img_processing(img,model):
      #print("image shape is: ", img.shape)
      img = cv.resize(img,(28,28),interpolation = cv.INTER_AREA)
      img = np.array(img, dtype=np.float32)
      #print(img[0])
--    #!it's super important to flip greyscale color since trained data is black background and white foreground
      img = 255 - img
      #print(img[0])
      cv.imwrite('app/util/tf_fashion_mnist/checkpoints/test.jpg',img)
--    if model == 'torch':
      ¦   return img
      elif model == 'tf':
      ¦   testimg = np.reshape(img,(1,28,28,1))
      #img = tf.image.convert_image_dtype(img, dtype=tf.float32)
      #img = tf.expand_dims(img,-1)
      #testimg = tf.reshape(img,(1,28,28,1))

      #print("test image size is: ", testimg.shape)
      return testimg
