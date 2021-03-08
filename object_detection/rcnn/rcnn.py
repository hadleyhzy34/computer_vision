import os,cv2,keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#image preprocessing
def img_preprocessing(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print('img cannot be read')
        return
    img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
    img = np.array(img, dtype=np.float32)
    return img

##test for image preprocessing
# img_path = '.\\data\\messi5.jpg'
# print(os.getcwd())
# img = img_preprocessing(img_path)
# cv2.imwrite('.\\data\\messi5_resize.jpg',img)

#alexnet
def create_alexnet(num_classes):
    """
    alexnet model
    """
    network = input_data(shape=[None, 224, 224, 3])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network

#pip3 install opencv-python == 3.4.2.16
# pip3 install opencv-contrib-python == 3.4.2.16
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()