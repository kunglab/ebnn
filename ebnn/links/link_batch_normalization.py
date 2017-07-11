from __future__ import absolute_import

import chainer.links as L


class BatchNormalization(L.BatchNormalization):
    def __init__(self, size, **kwargs):
        super(BatchNormalization, self).__init__(size, **kwargs)
        self.cname = "l_bn"
