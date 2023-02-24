#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index.
    """
    # 1. IID with 10 client
    # num_items = int(len(dataset)/num_users)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # print(dict_users)
    # print(all_idxs)
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])
    # return dict_users

    # 2. IID with 11 client, client 10 contains p1 = 0.1 total data
    # p1 = 0.5
    # num_p1 = int(len(dataset)*p1)
    # dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # dict_users[10] = set(np.random.choice(all_idxs, num_p1, replace=False))
    # all_idxs = list(set(all_idxs) - dict_users[10])
    #
    # num_items = int((len(dataset)-num_p1)/num_users)
    # for i in range(num_users):
    #     dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    #     all_idxs = list(set(all_idxs) - dict_users[i])
    # return dict_users

    # 3. NON-IID with 11 client, client 10 contains p1 = 0.1 total data
    lable_ind_0 = []
    lable_ind_1 = []
    lable_ind_2 = []
    lable_ind_3 = []
    lable_ind_4 = []
    lable_ind_5 = []
    lable_ind_6 = []
    lable_ind_7 = []
    lable_ind_8 = []
    lable_ind_9 = []
    for i in range(len(dataset)):
        image, label = dataset[i]
        if label == 0:
            lable_ind_0.append(i)
        elif label == 1:
            lable_ind_1.append(i)
        elif label == 2:
            lable_ind_2.append(i)
        elif label == 3:
            lable_ind_3.append(i)
        elif label == 4:
            lable_ind_4.append(i)
        elif label == 5:
            lable_ind_5.append(i)
        elif label == 6:
            lable_ind_6.append(i)
        elif label == 7:
            lable_ind_7.append(i)
        elif label == 8:
            lable_ind_8.append(i)
        else:
            lable_ind_9.append(i)

    dict_users = {}

    # # 3.1 all client
    # dict_users[0] = set(lable_ind_0)
    # dict_users[1] = set(lable_ind_1)
    # dict_users[2] = set(lable_ind_2)
    # dict_users[3] = set(lable_ind_3)
    # dict_users[4] = set(lable_ind_4)
    # dict_users[5] = set(lable_ind_5)
    # dict_users[6] = set(lable_ind_6)
    # dict_users[7] = set(lable_ind_7)
    # dict_users[8] = set(lable_ind_8)
    # dict_users[9] = set(lable_ind_9)

    # 3.2 all client + extra
    p1 = 0.5
    num_p1 = int(len(dataset)/10*p1)
    dict_users[10] = set(lable_ind_0[0:num_p1]+lable_ind_1[0:num_p1]+lable_ind_2[0:num_p1]+lable_ind_3[0:num_p1]
                         +lable_ind_4[0:num_p1]+lable_ind_5[0:num_p1]+lable_ind_6[0:num_p1]+lable_ind_7[0:num_p1]
                         +lable_ind_8[0:num_p1]+lable_ind_9[0:num_p1])

    # # all client
    dict_users[0] = set(lable_ind_0[num_p1::])
    dict_users[1] = set(lable_ind_1[num_p1+1::])
    dict_users[2] = set(lable_ind_2[num_p1+1::])
    dict_users[3] = set(lable_ind_3[num_p1+1::])
    dict_users[4] = set(lable_ind_4[num_p1+1::])
    dict_users[5] = set(lable_ind_5[num_p1+1::])
    dict_users[6] = set(lable_ind_6[num_p1+1::])
    dict_users[7] = set(lable_ind_7[num_p1+1::])
    dict_users[8] = set(lable_ind_8[num_p1+1::])
    dict_users[9] = set(lable_ind_9[num_p1+1::])

    return dict_users



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
