from __future__ import print_function
import paddle
import paddle.fluid as fluid
from vgg import *

def convolutional_neural_network():
    img = fluid.layers.data(
        name='img', shape=[1,64,64],dtype = 'float32')

#    net = VGG11()
#    return net.net(img)

#    conv1 = fluid.layers.conv2d(input=img, num_filters=20, filter_size=5, act='relu')
#    pool1 = fluid.layers.pool2d(input=conv1, pool_size=2, pool_stride=2, pool_type='max')
    h1 = fluid.nets.simple_img_conv_pool(input=img, num_filters=20, filter_size=5, act='relu', pool_size=2, pool_stride=2, pool_padding=0)
    b1 = fluid.layers.batch_norm(input=h1)
    
#    conv2 = fluid.layers.conv2d(input=pool1, num_filters=50, filter_size=5, act='relu')
#    pool2 = fluid.layers.pool2d(input=conv2, pool_size=2, pool_stride=2, pool_type='max')
    h2 = fluid.nets.simple_img_conv_pool(input=b1, num_filters=50, filter_size=5, act='relu', pool_size=2, pool_stride=2, pool_padding=0)
    
#    predict = fluid.layers.fc(input=pool2, size=6, act='softmax')
    predict = fluid.layers.fc(input=h2, size=6, act='softmax')
    return predict
