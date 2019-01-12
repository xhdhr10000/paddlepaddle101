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


#use_cuda = False
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

def train_func():
    label = fluid.layers.data(name='label', shape = [1],dtype = 'int64')
#     predict = softmax_regression()
#     predict = multilayer_perceptron()
    predict = convolutional_neural_network()

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost

def optimizer_func():
    optimizer = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
    return optimizer

feed_order = ['img', 'label'] 
params_dirname = "./model"

step = 0
epoch = 0
def event_handler_plot(event):
    global step
    if isinstance(event, EndStepEvent):
        print('Epoch %d step %d, loss %f' % (event.epoch, event.step, event.metrics[0]))
        if event.step % 20 == 0:
            test_metrics = trainer.test(
            reader=test_reader, feed_order=feed_order)
            print('Test %d, loss %f' % (event.step // 20, test_metrics[0]))

            if params_dirname is not None:
                trainer.save_params(params_dirname)

#             if test_metrics[0] < 1.0:
#                 print('loss is less than 10.0, stop')
#                 trainer.stop()
        step += 1

exe = fluid.Executor(place)
exe.run( fluid.default_startup_program() )

BATCH_SIZE = 128

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), 
        buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.test(), 
        buf_size=500),
    batch_size=BATCH_SIZE)

trainer = Trainer(
    train_func= train_func,
    place= place,
    optimizer_func= optimizer_func)

trainer.train(
    reader=train_reader,
    num_epochs=3,
    event_handler=event_handler_plot,
    feed_order=feed_order)
