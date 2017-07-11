import numpy as np

from chainer import link


class CLink(object):
    def generate_c(self):
        raise NotImplementedError(
            "Not implemented. This link cannot be exported as c.")

    def param_mem(self):
        raise NotImplementedError(
            "Not implemented. This link cannot be exported as c.")

    def temp_mem(self):
        raise NotImplementedError(
            "Not implemented. This link cannot be exported as c.")


class CChainMixin(object):
    def link_order(self):
        raise NotImplementedError(
            "Not implemented. This link cannot be exported as c.")

    def generate_c(self, filename, shape):
        self.to_cpu()
        h = np.random.random([1] + list(shape)).astype(np.float32)
        links = self.link_order()
        text = '#include "ebnn.h"\n'
        input_size = h.size
        inter_sizes = []
        for i, link in enumerate(links):
            inter_sizes.append(link.temp_mem(h.shape))
            text += link.generate_c(i, h.shape)
            h = link(h)

        inter_size = int(np.max(inter_sizes))

        text += """
uint8_t temp1[{inter_size}] = {{0}};
uint8_t temp2[{inter_size}] = {{0}};
""".format(input_size=input_size, inter_size=inter_size)
        text += "void ebnn_compute(float *input, uint8_t *output){\n"

        link_idx = 0
        link = links[0]
        text += "  {name}(input, temp1);\n".format(name=link.cname +
                                                   str(link_idx))
        for link in links[1:-1]:
            link_idx += 1
            if link_idx % 2 == 1:
                text += "  {name}(temp1, temp2);\n".format(
                    name=link.cname + str(link_idx))
            else:
                text += "  {name}(temp2, temp1);\n".format(
                    name=link.cname + str(link_idx))

        link_idx = len(links) - 1
        link = links[-1]
        if link_idx % 2 == 1:
            text += "  {name}(temp1, output);\n".format(name=link.cname + str(link_idx))
        else:
            text += "  {name}(temp2, output);\n".format(name=link.cname + str(link_idx))
        text += "}"

        with open(filename, 'w+') as fp:
            fp.write(text)


from link_bst import BST
from link_pool import Pool2D
from link_batch_normalization import BatchNormalization
from link_binary_convolution import BinaryConvolution2D
from link_binary_linear import BinaryLinear
from link_softmax_cross_entropy import SoftmaxCrossEntropy
from link_linear_BN_BST import LinearBNBST
from link_binary_linear_BN_BST import BinaryLinearBNBST
from link_binary_linear_softmax_layer import BinaryLinearSoftmax
from link_binary_linear_BN_softmax_layer import BinaryLinearBNSoftmax
from link_conv_BN_BST import ConvBNBST
from link_binary_conv_BN_BST import BinaryConvBNBST
from link_conv_pool_BN_BST import ConvPoolBNBST
from link_binary_conv_pool_BN_BST import BinaryConvPoolBNBST
