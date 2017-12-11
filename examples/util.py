import os
import argparse

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import training
from chainer.training import extensions
from chainer.serializers import load_hdf5, save_hdf5
from chainer.datasets import get_mnist, get_cifar10
import numpy as np


def accuracy(model, dataset_tuple, ret_param='acc', batchsize=1024, gpu=0):
    xp = np if gpu < 0 else cuda.cupy
    x, y = dataset_tuple._datasets[0], dataset_tuple._datasets[1]
    accs = 0
    model.train = False
    for i in range(0, len(x), batchsize):
        x_batch = xp.array(x[i:i + batchsize])
        y_batch = xp.array(y[i:i + batchsize])
        acc_data = model(x_batch, y_batch, ret_param=ret_param)
        acc_data.to_cpu()
        acc = acc_data.data
        accs += acc * len(x_batch)
    return (accs / len(x)) * 100.


def train_model(model, train, test, args, lr=0.003):
    chainer.config.train = True
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    opt = optimizers.Adam(lr)
    opt.setup(model)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, opt, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.LogReport())
    if args.verbose:
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss'] +
            model.report_params() +
            ['elapsed_time']))
        trainer.extend(extensions.ProgressBar())
    trainer.run()

    name = model.param_names()
    save_model(model, os.path.join(args.model_path, name))
    with open(os.path.join(args.out, 'log'), 'r') as fp:
        return eval(fp.read().replace('\n', ''))
    chainer.config.train = False


def save_model(model, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_hdf5(os.path.join(folder, 'model.hdf5'), model)
    return model


def load_model(model, folder, gpu=0):
    load_hdf5(os.path.join(folder, 'model.hdf5'), model)
    if gpu >= 0:
        model = model.to_gpu(gpu)
    return model


def load_or_train_model(model, train, test, args):
    name = model.param_names()
    model_folder = os.path.join(args.model_path, name)
    if not os.path.exists(model_folder) or args.overwrite_models:
        train_model(model, train, test, args)
    else:
        load_model(model, model_folder, gpu=args.gpu)


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return get_mnist(ndim=3)
    if dataset_name == 'cifar10':
        return get_cifar10(ndim=3)
    raise NameError('{}'.format(dataset_name))


def default_parser(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dataset', '-d', default='mnist',
                        choices=['mnist', 'cifar10'], help='dataset name')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='_output',
                        help='Directory to output the result')
    parser.add_argument('--model_path', default='_models/',
                        help='Directory to store the models (for later use)')
    parser.add_argument('--overwrite_models', action='store_true',
                        help='If true, reruns a setting and overwrites old models')
    parser.add_argument('--verbose', action='store_true',
                        help='Prints output per iteration')
    return parser
