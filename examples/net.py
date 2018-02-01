import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import chainer
import chainer.functions as F
from chainer import reporter

import ebnn.links as BL


class ConvNet(BL.CChainMixin, chainer.Chain):
    def __init__(self, n_in, n_out, n_filters):
        super(ConvNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_filters = n_filters
        with self.init_scope():
            self.l1 = BL.ConvPoolBNBST(n_in, n_filters, 3, stride=2, pksize=3, pstride=2)
            self.l2 = BL.BinaryLinearBNSoftmax(None, n_out)

    def link_order(self):
        return [self.l1,  self.l2]

    def __call__(self, x, t, ret_param='loss'):
        h = self.l1(x)
        h = self.l2(h)

        # reports loss and accuracy (used during training)
        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    # Determines what is output each iteration during training
    # (shouldn't need to change this unless your network is complex)
    def report_params(self):
        return ['validation/main/acc']

    # A unique identifier of the model (used to save models)
    # I prefer this over random generated ids as its readable
    def param_names(self):
        # in this case: n_filters, n_in and n_out define the model
        return 'ConvNet{}_{}_{}'.format(self.n_filters, self.n_in, self.n_out)


class BinaryConvNet(BL.CChainMixin, chainer.Chain):
    '''
    ConvNet where input in Binary
    '''
    def __init__(self, n_in, n_out, n_filters):
        super(BinaryConvNet, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_filters = n_filters
        with self.init_scope():
            self.l1 = BL.BinaryConvPoolBNBST(n_in, n_filters, 3, stride=2, pksize=3, pstride=2)
            self.l2 = BL.BinaryLinearBNSoftmax(None, n_out)

    def link_order(self):
        return [self.l1,  self.l2]

    def __call__(self, x, t, ret_param='loss'):
        h = self.l1(x)
        h = self.l2(h)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'BinaryConvNet{}_{}_{}'.format(self.n_filters, self.n_in, self.n_out)


class AlexNet(BL.CChainMixin, chainer.Chain):
    def __init__(self, n_in, n_out):
        super(AlexNet, self).__init__()
        self.n_out = n_out
        with self.init_scope():
            self.conv1 = BL.ConvPoolBNBST(n_in, 10, 5, stride=1, pad=0, pksize=3, pstride=2)
            self.conv2 = BL.BinaryConvPoolBNBST(10, 10, 5, stride=1, pad=2, pksize=3, pstride=2)
            self.conv3 = BL.BinaryConvBNBST(10, 10, 3, stride=1, pad=1)
            self.conv4 = BL.BinaryConvBNBST(10, 10, 3, stride=1, pad=1)
            self.conv5 = BL.BinaryConvPoolBNBST(10, 10, 3, stride=1, pad=1, pksize=3, pstride=2)
            self.fc8 = BL.BinaryLinearBNSoftmax(40, n_out)

    def link_order(self):
        return [self.conv1,
                self.conv2,
                self.conv3,
                self.conv4,
                self.conv5,
                self.fc8]

    def __call__(self, x, t, ret_param='loss'):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.fc8(h)

        report = {
            'loss': F.softmax_cross_entropy(h, t),
            'acc': F.accuracy(h, t)
        }

        reporter.report(report, self)
        return report[ret_param]

    def report_params(self):
        return ['validation/main/acc']

    def param_names(self):
        return 'AlexNet{}'.format(self.n_in, self.n_out)