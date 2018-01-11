import numpy as np

import util
import net

if __name__ == '__main__':
    parser = util.default_parser('MLP Example')
    args = parser.parse_args()

    # get the dataset (default is MNIST)
    train, test = util.get_dataset(args.dataset)

    n_in = train._datasets[0].shape[1]
    n_out = len(np.unique(train._datasets[1]))

    # initialize model
    model = net.ConvNet(n_in, n_out, n_filters=10)

    # train model
    util.train_model(model, train, test, args)

    # get test accuracy
    acc = util.accuracy(model, test, gpu=args.gpu)
    print('Model accuracy: ', acc)

    # generate and save C model as a header file
    model.generate_c('simple_{}.h'.format(args.dataset), train._datasets[0].shape[1:])
