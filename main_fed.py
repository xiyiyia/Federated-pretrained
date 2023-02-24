#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
import pandas as pd


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    test_acc = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    print("===Start pre-training===")
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[10])
    w, loss = local.train(net=net_glob.to(args.device), num_iter=150)

    print("===Start FL-training===")
    for iter in range(150):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        idxs_users = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        print("Clients:"+str(idxs_users))
        for idx in idxs_users:
            # print(len(dict_users[idx]))
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), num_iter=args.local_ep)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print("===Start in-training===")
        # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[10])
        # w, loss = local.train(net=net_glob.to(args.device), num_iter=args.local_ep)

        # print loss and test acc
        loss_avg = sum(loss_locals) / len(loss_locals)
        net_glob.eval()
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print('Round {:3d}, Average loss {:.3f}, Test acc {:.3f}'.format(iter, loss_avg, acc_test))
        test_acc.append(acc_test.numpy())

    # print("===Start poster-training===")
    # for iter in range(50):
    #     # print(len(dict_users[10]))
    #     local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[10])
    #     w, loss = local.train(net=net_glob.to(args.device), num_iter=args.local_ep)
    #     net_glob.eval()
    #     acc_test, loss_test = test_img(net_glob, dataset_test, args)
    #     print('Round {:3d}, Average loss {:.3f}, Test acc {:.3f}'.format(iter, loss_avg, acc_test))
    #     test_acc.append(acc_test.numpy())

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(test_acc)), test_acc)
    # plt.ylabel('test_acc')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    df = pd.DataFrame(data=test_acc)
    df.to_csv("./save/NON-IID_0.5_3_preFL.csv", encoding='utf-8', index=False)

