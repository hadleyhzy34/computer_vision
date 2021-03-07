import numpy as np
import tensorflow as tf

#basic convolution process
x = np.array([[1,2,3,7],[6,3,4,2]])
k = np.array([[-0.0308,-0.4722],[-0.3021,-0.0464]])
tensor_x = tf.constant(x,tf.float32)
tensor_k = tf.constant(k,tf.float32)
tensor_res = tf.nn.convolution(tf.reshape(tensor_x,[1,2,4,1]),tf.reshape(tensor_k,[2,2,1,1]), strides=2, padding='VALID')
print(tensor_res)

