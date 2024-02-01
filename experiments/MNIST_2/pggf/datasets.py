import numpy as np
import torch
# import os
# from os import path
# from sklearn.model_selection import KFold
import pandas as pd
# import zipfile
# import urllib.request
import torchvision



class UCIDatasets():
    def __init__(self, name, data_path="", device=None):
        self.name = name
        self.data_path = data_path
        self.device = device
        self._load_dataset()

    def _load_dataset(self):
        partion = 0.5
        if self.name == "sonar":
            data = pd.read_csv(self.data_path + '/sonar.csv', header=0, delimiter=",").values

            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion*data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = torch.tensor(train_y).to(torch.int64).to(self.device)
            self.test_y = torch.tensor(test_y).to(torch.int64).to(self.device)
            out_dim=2
        elif self.name == "wine_red":
            data = pd.read_csv(self.data_path + '/winequality-red.csv', header=0, delimiter=";").values

            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = torch.tensor(train_y).to(torch.int64).to(self.device) - 1
            self.test_y = torch.tensor(test_y).to(torch.int64).to(self.device) - 1
            out_dim = 10
        elif self.name == "wine_white":
            data = pd.read_csv(self.data_path + '/winequality-white.csv', header=0, delimiter=";").values

            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = torch.tensor(train_y).to(torch.int64).to(self.device) - 1
            self.test_y = torch.tensor(test_y).to(torch.int64).to(self.device) - 1
            out_dim = 10
        elif self.name == "heart":
            data = pd.read_csv(self.data_path + '/processed.cleveland.data', header=0, delimiter=",").values
            data = data[np.random.permutation(np.arange(len(data)))]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[i, j] == "?":
                        data[i, j] = 0
            data = data.astype(float)
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = (torch.tensor(train_y) < 0.5).to(torch.int64).to(self.device)
            self.test_y = (torch.tensor(test_y) > 0.5).to(torch.int64).to(self.device)
            out_dim = 2

        elif self.name == "glass":
            data = pd.read_csv(self.data_path + '/glass.data', header=0, delimiter=",").values
            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, 1:-1]
            test_x = data[train_num:, 1:-1]
            train_y = data[:train_num, -1] - 1
            test_y = data[train_num:, -1] - 1
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = (torch.tensor(train_y)).to(torch.int64).to(self.device)
            self.test_y = (torch.tensor(test_y)).to(torch.int64).to(self.device)
            out_dim = 7
        elif self.name == "german":
            data = pd.read_csv(self.data_path + '/german.data-processed', header=0, delimiter=" ").values[:, :-2]

            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = torch.tensor(train_y).to(torch.int64).to(self.device) - 1
            self.test_y = torch.tensor(test_y).to(torch.int64).to(self.device) - 1
            out_dim = 2
        elif self.name == "australian":
            data = pd.read_csv(self.data_path + '/australian.dat', header=0, delimiter=" ").values

            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1].astype(float)
            test_x = data[train_num:, :-1].astype(float)
            train_y = data[:train_num, -1].astype(int)
            test_y = data[train_num:, -1].astype(int)
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = torch.tensor(train_y).to(torch.int64).to(self.device)
            self.test_y = torch.tensor(test_y).to(torch.int64).to(self.device)
            out_dim = 2
        elif self.name == "covertype":
            data = pd.read_csv(self.data_path + '/covtype.data', header=0, delimiter=",").values
            data = data[np.random.permutation(np.arange(len(data)))[:8000]]

            data = data.astype(float)
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, :-1]
            test_x = data[train_num:, :-1]
            train_y = data[:train_num, -1]
            test_y = data[train_num:, -1]
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            x_stds[x_stds==0] = 1.
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = (torch.tensor(train_y)-1).to(torch.int64).to(self.device)
            self.test_y = (torch.tensor(test_y)-1).to(torch.int64).to(self.device)
            out_dim = 7
        elif self.name == "glass":
            data = pd.read_csv(self.data_path + '/glass.data', header=0, delimiter=",").values
            data = data[np.random.permutation(np.arange(len(data)))]
            train_num = int(partion * data.shape[0])
            train_x = data[:train_num, 1:-1]
            test_x = data[train_num:, 1:-1]
            train_y = data[:train_num, -1] - 1
            test_y = data[train_num:, -1] - 1
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            self.train_x = torch.from_numpy((train_x - x_means) / x_stds).to(self.device)
            self.test_x = torch.from_numpy((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = (torch.tensor(train_y)).to(torch.int64).to(self.device)
            self.test_y = (torch.tensor(test_y)).to(torch.int64).to(self.device)
            out_dim = 7
        elif self.name == "mnist":
            data_train = torchvision.datasets.MNIST(self.data_path, download=True, train=True)
            data_test = torchvision.datasets.MNIST(self.data_path, download=True, train=False)

            train_x = data_train.data.reshape(data_train.data.shape[0], -1).float()
            test_x = data_test.data.reshape(data_test.data.shape[0], -1).float()
            test_x = test_x + torch.randn_like(test_x)
            train_y = data_train.targets
            test_y = data_test.targets
            #
            # print(self.device)
            x_means, x_stds = train_x.mean(axis=0), train_x.var(axis=0) ** 0.5
            x_stds[x_stds == 0] = 1.
            self.train_x = ((train_x - x_means) / x_stds).to(self.device)
            self.test_x = ((test_x - x_means) / x_stds).to(self.device)
            # self.train_y = torch.nn.functional.one_hot(torch.tensor(train_y).to(torch.int64)).float()
            # self.test_y = torch.nn.functional.one_hot(torch.tensor(test_y).to(torch.int64)).float()
            self.train_y = train_y.to(torch.int64).to(self.device)
            self.test_y = test_y.to(torch.int64).to(self.device)
            out_dim = 10

        self.in_dim = self.test_x.shape[1]
        self.out_dim = out_dim
    #
    # def get_split(self, split=-1, train=True):
    #     if split == -1:
    #         split = 0
    #     if 0 <= split and split < self.n_splits:
    #         train_index, test_index = self.data_splits[split]
    #         x_train, y_train = self.data[train_index,
    #                            :self.in_dim], self.data[train_index, self.in_dim:]
    #         x_test, y_test = self.data[test_index, :self.in_dim], self.data[test_index, self.in_dim:]
    #         x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
    #         y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5
    #         x_train = (x_train - x_means) / x_stds
    #         y_train = (y_train - y_means) / y_stds
    #         x_test = (x_test - x_means) / x_stds
    #         y_test = (y_test - y_means) / y_stds
    #         if train:
    #             inps = torch.from_numpy(x_train).float()
    #             tgts = torch.from_numpy(y_train).float()
    #             train_data = torch.utils.data.TensorDataset(inps, tgts)
    #             return train_data
    #         else:
    #             inps = torch.from_numpy(x_test).float()
    #             tgts = torch.from_numpy(y_test).float()
    #             test_data = torch.utils.data.TensorDataset(inps, tgts)
    #             return test_data