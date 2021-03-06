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
    im = im.resize((64, 64), Image.ANTIALIAS)
    im = np.array(im).astype(np.float32) / 255.0
    im = 1 - im
    im = (im - np.mean(im)) / np.std(im)
    out = Image.fromarray(im * 255).convert('RGB')
    out.save('out.png')
    return im.reshape(1, 1, 64, 64)

path = 'dataset/1.png'
if len(sys.argv) > 1:
    path = sys.argv[1]
img = load_image(path)

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

print('input shape: {}'.format(img.shape))
results = inferencer.infer({'img': img})
lab = np.argsort(results)  # probs and lab are the results of one batch data
print("Label of %s is: %d" % (path, lab[0][0][-1]))
