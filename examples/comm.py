import os
import sys
sys.path.append('..')

import chainer
from deepopt.trainer import Trainer
from chainer_sequential.sequential import Sequential
from chainer_sequential.chain import Chain
from chainer_sequential.function import *
from chainer_sequential.link import *
from chainer_sequential.binary_link import *
from chainer_sequential.binary.links import CLink

class CommDropoutB(chainer.Chain):
    def __init__(self, ratio=0.5):
        self.ratio = 0.5
        self.cname = "l_b_linear_softmax"
        super(CommDropoutB, self).__init__()

    def __call__(self, x, test=False):
        return chainer.functions.dropout(x, self.ratio)

class CommDropout(BinaryLink):
    def __init__(self, in_channels, out_channels, ratio=0.5):
        self._link = "CommDropout"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio

    def to_link(self):
        args = self.to_chainer_args()
        print(args)
        return CommDropoutB(**args)


input_dims = 1
output_dims = 10
train, test = chainer.datasets.get_mnist(ndim=3)
nfilters = 64
nepochs = 16
path = "/home/brad/data/data/comm"
path += "_{}".format(nfilters)

model = Sequential()
model.add(ConvPoolBNBST(input_dims, nfilters, 3, 1, 1, 3, 1, 1))
model.add(CommDropout(nfilters, nfilters))
model.add(BinaryConvPoolBNBST(nfilters, nfilters, 3, 1, 1, 3, 1, 1))
model.add(BinaryLinearBNSoftmax(None, output_dims))
model.build()

chain = Chain()
chain.add_sequence(model)
chain.setup_optimizers('adam', 0.001)

trainer = Trainer(path, chain, train, test, nepoch=nepochs)
res = trainer.run()
