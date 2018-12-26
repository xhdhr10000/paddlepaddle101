from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import os
from PIL import Image
import sys
from net import convolutional_neural_network
from data import read_data
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print("In the fluid 1.0, the trainer and inference are moving to paddle.fluid.contrib", file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *

use_cuda = False
# use_cuda = True
place = fluid.CUDAPlace(1) if use_cuda else fluid.CPUPlace()
params_dirname = "./model"

inferencer = Inferencer(
#     infer_func=softmax_regression, # uncomment for softmax regression
#     infer_func=multilayer_perceptron, # uncomment for MLP
    infer_func=convolutional_neural_network,  # uncomment for LeNet5
    param_path=params_dirname,
    place=place )

count = 0
bingo = 0
for data in read_data('test')():
    count = count + 1
    
    img = data[0]
    label = int(data[1])
    result = inferencer.infer({'img': img})
    lab = np.argmax(result)
    prob = np.max(result)
    print('Test case %d: label %d predict %d with probability %f' % (count, label, lab, prob))

    if label == lab:
        bingo = bingo + 1
    if count > 100:
        break
print('Total count: %d/%d, rate %f' % (bingo, count, float(bingo) / count))
