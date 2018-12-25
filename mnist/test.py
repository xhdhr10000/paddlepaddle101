from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
import os
from PIL import Image
import sys
from net import convolutional_neural_network
try:
    from paddle.fluid.contrib.trainer import *
    from paddle.fluid.contrib.inferencer import *
except ImportError:
    print("In the fluid 1.0, the trainer and inference are moving to paddle.fluid.contrib", file=sys.stderr)
    from paddle.fluid.trainer import *
    from paddle.fluid.inferencer import *


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = im / 255.0 * 2.0 - 1.0
    return im

cur_dir = os.getcwd()
img = load_image(cur_dir + '/image/infer_3.png')

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

results = inferencer.infer({'img': img})
lab = np.argsort(results)  # probs and lab are the results of one batch data
print("Label of image/infer_3.png is: %d" % lab[0][0][-1])

count = 0
bingo = 0
for data in paddle.dataset.mnist.test()():
    count = count + 1
    
    img = data[0]
    label = data[1]
    img = np.array(img).reshape(1, 1, 28, 28).astype(np.float32)
    result = inferencer.infer({'img': img})
    lab = np.argmax(result)
    prob = np.max(result)
    print('Test case %d: label %d predict %d with probability %f' % (count, label, lab, prob))

    if label == lab:
        bingo = bingo + 1
#    if count > 2000:
#        break
print('Total count: %d/%d, rate %f' % (bingo, count, float(bingo) / count))
