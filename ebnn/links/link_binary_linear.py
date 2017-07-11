from __future__ import absolute_import

import numpy
from chainer import link

from ..functions import function_binary_linear
import math
from chainer import initializers
from chainer import cuda
from chainer import variable

class BinaryLinear(link.Link):
    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(BinaryLinear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return function_binary_linear.binary_linear(x, self.W, self.b)