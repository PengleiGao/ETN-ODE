import numpy as np
import pandas as pd
import torch

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def getGenerator(data_name):
    """
    Arguments:
        data_name    - (string) name of data file
    Returns:
        DataGenerator class that fits data_name
    """

    return GeneralGenerator

class subDataset(torch.utils.data.Dataset):
    def __init__(self, X_input, X_target):
        super(subDataset, self).__init__()
        self.X_input = X_input
        self.X_target = X_target
    def __len__(self):
        return (self.X_input).shape[0]
    def __getitem__(self, idx):
        return (self.X_input[idx], self.X_target[idx])


class DataGenerator():
    def __init__(self, data, target_col, indep_col, win_size, pre_T, train_share, is_stateful=False, normalize_pattern=2):
        self.data = data
        self.train_share = train_share
        self.val_num = train_share + (1 - train_share) / 2
        self.win_size = win_size
        self.pre_T = pre_T
        self.target_col = target_col
        self.indep_col = indep_col
        self.normalize_pattern = normalize_pattern
        self.is_stateful = is_stateful

    def normalize(self, X):
        def _norm_col(arr):
            stat = np.zeros((2,))
            stat[0] = arr.min()
            stat[1] = arr.max()
            if abs(stat[0] - stat[1]) < 1e-10:
                pass
            else:
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            return arr, stat

        if self.normalize_pattern == 0:
            pass
        # elif self.normalize_pattern == 1:
        #     n_dim = X.shape[-1]
        #     stats = np.zeros((2, n_dim))
        #     if np.ndim(X) == 2:
        #         for d in range(n_dim):
        #             X[:, d], stats[:, d] = _norm_col(X[:, d])
        #     elif np.ndim(X) == 3:
        #         for d in range(n_dim):
        #             X[:, :, d], stats[:, d] = _norm_col(X[:, :, d])
        elif self.normalize_pattern == 2:
            #n_dim = X.shape[-1]
            #stats = np.zeros((2,))
            means = np.mean(X, axis=0, dtype=np.float32)
            stds = np.std(X, axis=0, dtype=np.float32)
            # stats[0] = means
            # stats[1] = stds
            X = (X - means) / (stds + (stds == 0) * .001)
        else:
            raise Exception('invalid normalize_pattern')
        return X.astype(np.float32), means, stds

    def with_target(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_test = (self.data[n_train:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        dta_target_test = self.data[n_train:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size, n_tr):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                #tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

            for j in range(self.win_size, n_te):
                te_x = dta_test[j - self.win_size:j]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, X_test, Y_train, Y_test, ymean, ystd

    def with_target2(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_val = (self.data[n_train:n_val] - meanx) / stdx
        dta_test = (self.data[n_train:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_train:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_val = len(dta_val) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_val = []
        X_test = []
        Y_train = []
        Y_val = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size, n_tr):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

            for k in range(self.win_size, n_val):
                val_x = dta_val[k - self.win_size:k]
                X_val.append(val_x)
                val_y = dta_target_val[k - self.win_size:k + self.pre_T]
                #val_y = dta_target_val[k:k + self.pre_T]
                Y_val.append(val_y)

            for j in range(self.win_size, n_te):
                te_x = dta_test[j - self.win_size:j]
                X_test.append(te_x)
                te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                #te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, ymean, ystd

    def getdata(self):
        dta_target = self.data[:, self.target_col]
        dta_x_norm, _, _ = self.normalize(self.data)
        #norm_y, y_mean, y_std = self.normalize(dta_target)
        n = len(self.data) - self.pre_T
        if n < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_all = []
        Y_all = []

        if self.is_stateful:
            for i in range(self.win_size, n, self.win_size):
                tmx = dta_x_norm[i - self.win_size:i]

                X_all.append(tmx)
                Y_all.append(dta_target[i])
        else:
            for i in range(self.win_size, n):
                tmx = dta_x_norm[i - self.win_size:i]
                X_all.append(tmx)
                #tmy = dta_target[i - self.win_size:i + self.pre_T]
                tmy = dta_target[i:i + self.pre_T]
                # tmy = np.expand_dims(list_target[i:i+5], axis=1)
                Y_all.append(tmy)

        X_all = np.array(X_all)
        Y_all = np.array(Y_all)
        row_num = X_all.shape[0]
        n_train = int(row_num * self.train_share)
        # n_valid = int(self.row_num * (train_share[0] + train_share[1]))
        n_test = row_num
        X_train = X_all[:n_train]
        # X_valid = X_all[n_train:n_valid]
        X_test = X_all[n_train:n_test]
        Y_train = Y_all[:n_train]
        Y_test = Y_all[n_train:n_test]
        norm_y, y_mean, y_std = self.normalize(Y_train)

        return X_train, X_test, Y_train, Y_test, y_mean, y_std

    def with_target3(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_test = (self.data[n_train:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_test = self.data[n_train:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size * 2, n_tr, 2):
                tr_x = dta_train[i - self.win_size * 2:i]
                tr_x = tr_x[::2]
                X_train.append(tr_x)
                #tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                tr_y = dta_target_train[i:i + self.pre_T]
                tr_y = tr_y[::2]
                Y_train.append(tr_y)

            for j in range(self.win_size * 2, n_te, 2):
                te_x = dta_test[j - self.win_size * 2:j]
                te_x = te_x[::2]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        #norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, X_test, Y_train, Y_test, mean_y, std_y

    def with_target_latent(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_test = (self.data[n_train:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_test = self.data[n_train:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_test = []
        Y_train = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size * 2, n_tr, 2):
                tr_x = dta_train[i - self.win_size * 2:i]
                tr_x = tr_x[::2]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size * 2:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                tr_y = tr_y[::2]
                Y_train.append(tr_y)

            for j in range(self.win_size * 2, n_te, 2):
                te_x = dta_test[j - self.win_size * 2:j]
                te_x = te_x[::2]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size * 2:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        #norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, X_test, Y_train, Y_test, mean_y, std_y

    def data_extract_val(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_val = (self.data[n_train:n_val] - meanx) / stdx
        dta_test = (self.data[n_val:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_val:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_val = len(dta_val) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_val = []
        X_test = []
        Y_train = []
        Y_val = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size, n_tr):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                #tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

            for k in range(self.win_size, n_val):
                val_x = dta_val[k - self.win_size:k]
                X_val.append(val_x)
                #val_y = dta_target_val[k - self.win_size:k + self.pre_T]
                val_y = dta_target_val[k:k + self.pre_T]
                Y_val.append(val_y)

            for j in range(self.win_size, n_te):
                te_x = dta_test[j - self.win_size:j]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, ymean, ystd

    def data_extract_val_arbi(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_val = (self.data[n_train:n_val] - meanx) / stdx
        dta_test = (self.data[n_val:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_val:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_val = len(dta_val) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_val = []
        X_test = []
        Y_train = []
        Y_val = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size * 2, n_tr, 2):
                tr_x = dta_train[i - self.win_size * 2:i]
                tr_x = tr_x[::2]
                X_train.append(tr_x)
                #tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                tr_y = dta_target_train[i:i + self.pre_T]
                tr_y = tr_y[::2]
                Y_train.append(tr_y)

            for k in range(self.win_size * 2, n_val, 2):
                val_x = dta_val[k - self.win_size * 2:k]
                val_x = val_x[::2]
                X_val.append(val_x)
                #val_y = dta_target_val[k - self.win_size:k + self.pre_T]
                val_y = dta_target_val[k:k + self.pre_T]
                Y_val.append(val_y)

            for j in range(self.win_size * 2, n_te, 2):
                te_x = dta_test[j - self.win_size * 2:j]
                te_x = te_x[::2]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        #norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean_y, std_y

    def data_extract_val_arbi_latent(self):
        row_num = self.data.shape[0]
        n_train = int(row_num * self.train_share)
        n_val = int(row_num * self.val_num)
        n_test = row_num
        dta_train, meanx, stdx = self.normalize(self.data[:n_train])
        #dta_test, _, _ = self.normalize(self.data[n_train:n_test])
        dta_val = (self.data[n_train:n_val] - meanx) / stdx
        dta_test = (self.data[n_val:n_test] - meanx) / stdx
        dta_target_train = self.data[:n_train, self.target_col]
        squeeze_y = dta_target_train[::2]
        norm_y, mean_y, std_y = self.normalize(squeeze_y)
        dta_target_val = self.data[n_train:n_val, self.target_col]
        dta_target_test = self.data[n_val:n_test, self.target_col]
        n_tr = len(dta_train) - self.pre_T
        n_val = len(dta_val) - self.pre_T
        n_te = len(dta_test) - self.pre_T
        if n_tr < self.win_size:
            print("\n ERROR: SIZE \n")
            return

        X_train = []
        X_val = []
        X_test = []
        Y_train = []
        Y_val = []
        Y_test = []

        if self.is_stateful:
            for i in range(self.win_size, n_tr, self.win_size):
                tr_x = dta_train[i - self.win_size:i]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                Y_train.append(tr_y)

        else:
            for i in range(self.win_size * 2, n_tr, 2):
                tr_x = dta_train[i - self.win_size * 2:i]
                tr_x = tr_x[::2]
                X_train.append(tr_x)
                tr_y = dta_target_train[i - self.win_size * 2:i + self.pre_T]
                #tr_y = dta_target_train[i:i + self.pre_T]
                tr_y = tr_y[::2]
                Y_train.append(tr_y)

            for k in range(self.win_size * 2, n_val, 2):
                val_x = dta_val[k - self.win_size * 2:k]
                val_x = val_x[::2]
                X_val.append(val_x)
                #val_y = dta_target_val[k - self.win_size:k + self.pre_T]
                val_y = dta_target_val[k:k + self.pre_T]
                Y_val.append(val_y)

            for j in range(self.win_size * 2, n_te, 2):
                te_x = dta_test[j - self.win_size * 2:j]
                te_x = te_x[::2]
                X_test.append(te_x)
                #te_y = dta_target_test[j - self.win_size:j + self.pre_T]
                te_y = dta_target_test[j:j + self.pre_T]
                Y_test.append(te_y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        #norm_y, ymean, ystd = self.normalize(Y_train)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test, mean_y, std_y

class GeneralGenerator(DataGenerator):
    def __init__(self, data_path, target_col, indep_col, win_size, pre_T, train_share=0.9, is_stateful=False, normalize_pattern=2):
        X = pd.read_csv(data_path, dtype=np.float32)
        super(GeneralGenerator, self).__init__(X.values,
                                               target_col=target_col,
                                               indep_col=indep_col,
                                               win_size=win_size,
                                               pre_T=pre_T,
                                               train_share=train_share,
                                               is_stateful=is_stateful,
                                               normalize_pattern=normalize_pattern)

class smlGenerator(DataGenerator):
    def __init__(self, data_path, target_col, indep_col, win_size, pre_T, train_share=0.9, is_stateful=False, normalize_pattern=2):
        X1 = np.loadtxt(data_path, delimiter=',')
        X2 = np.loadtxt(data_path, delimiter=',')
        X = np.concatenate(X1, X2, axis=0)
        super(smlGenerator, self).__init__(X,
                                           target_col=target_col,
                                           indep_col=indep_col,
                                           win_size=win_size,
                                           pre_T=pre_T,
                                           train_share=train_share,
                                           is_stateful=is_stateful,
                                           normalize_pattern=normalize_pattern)