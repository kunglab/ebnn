import util
import net

if __name__ == '__main__':
    parser = util.default_parser('MLP Example')
    args = parser.parse_args()

    # get the dataset (default is MNIST)
    train, test = util.get_dataset(args.dataset)

    # initialize model
    model = net.AlexNet(n_out=10)

    # train model
    util.train_model(model, train, test, args)

    # get test accuracy
    acc = util.accuracy(model, test, gpu=args.gpu)
    print('Model accuracy: ', acc)

    # generate and save C model as a header file
    model.generate_c('alex.h', train._datasets[0].shape[1:])
