import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import util
from ebnn.utils import binary_util


if __name__ == '__main__':
    parser = util.default_parser('MLP Example')
    args = parser.parse_args()

    # get the dataset (default is MNIST)
    train, test = util.get_dataset(args.dataset)

    x = train._datasets[0][0:20]
    if 'binary' in args.dataset:
        x[x==-1] = 0
        x = x.astype(np.uint8)
        x_str = binary_util.np_to_packed_uint8C(x, 'train_data', 'row_major')
    else:
        x_str = binary_util.np_to_floatC(x, 'train_data', 'row_major')

    y_str = binary_util.np_to_floatC(train._datasets[1][0:20], 'train_labels', 'row_major')

    with open(args.dataset, 'w+') as fp:
        fp.write(x_str)
        fp.write(y_str)
