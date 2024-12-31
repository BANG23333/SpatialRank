import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import seaborn as sns
import matplotlib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import os
import einops
from sklearn.metrics import accuracy_score
from ConvLSTM import *
from GC import GraphConv
from new_hint import HintNet
from scipy.ndimage import gaussian_filter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_all_seeds(SEED):
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


set_all_seeds(0)


def plot_loss(train_loss_arr, valid_loss_arr):
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(train_loss_arr, 'k', label='NDCG by ApproxNDCG')
    ax1.plot(valid_loss_arr, 'g', label='NDCG by CE')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    # ax2.plot(train_mape_arr, 'r--', label='train_mape_arr')
    # ax2.plot(v_mape_arr, 'b--', label='v_mape_arr')

    ax2.legend(loc=2)
    plt.show()
    plt.clf()

    return fig


def normalize(X):
    x_max = np.zeros(X.shape[3])
    x_min = np.zeros(X.shape[3])
    x_dif = np.zeros(X.shape[3])

    for i in range(X.shape[3]):
        x_max[i] = np.max(X[:, :, :, i])
        x_min[i] = np.min(X[:, :, :, i])

    x_dif = x_max - x_min

    x_dif = np.where(x_dif == 0, 1, x_dif)

    new_X = np.zeros(X.shape)

    for ctr in range(X.shape[0]):
        for x in range(X.shape[1]):
            for y in range(X.shape[2]):
                new_X[ctr][x][y] = (X[ctr][x][y] - x_min) / x_dif

    return new_X


def Discounted_Gain(y, ranking):
    nomi = 2 ** y - 1
    denomi = np.log2(2 + ranking)
    return nomi / denomi


def rank_with_ties(scores):
    foo_unique = np.unique(scores)
    ranking = np.argsort(foo_unique)[::-1]
    holder = {}
    for i, j in zip(foo_unique, ranking):
        holder[i] = j

    ranking = []
    for each in scores:
        ranking.append(holder[each])

    return np.array(ranking)


def DG_mapping(dif, true):
    out = np.zeros(dif.shape[1])

    for i, j in zip(dif, true):
        out += Discounted_Gain(i, rank_with_ties(j))

    return out / dif.shape[0]


"""
def spa_rank_with_ties(scores):

    foo_unique = np.unique(scores)
    ranking = np.argsort(foo_unique)[::-1]
    holder = {}

    for i, j in zip(foo_unique, ranking):
        holder[i] = j

    ranking = []
    for each in scores:
        ranking.append(holder[each])
    ranking = np.array(ranking)
    spa_rank = np.copy(ranking)
    for i in range(len(ranking)):
        for j in range(len(ranking)):
            if nei_dict[i][j] == 1:
                spa_rank[i] = min(ranking[i], ranking[j])

    return spa_rank

def spa_dcg(true, pred, k):

    out = np.zeros(true.shape[1])

    for true_list, pred_list in zip(true, pred):
        spa_true = true_list.copy()
        for i in range(len(true_list)):
            for j in range(len(true_list)):
                if nei_dict[i][j] == 1:
                    spa_true[i] = max(true_list[i], true_list[j])

        top_k_rank = np.argsort(pred_list)[::-1][k]
        temp = rank_with_ties(pred_list)
        dg = Discounted_Gain(spa_true, temp)
        out += np.where(temp<=top_k_rank, dg, 0)
    return np.sum(out/np.sum(temp<=top_k_rank))
"""


def spa_ndcg(true, pred, k):
    true = np.squeeze(map_squre_back(true))
    r_true = np.zeros(true.shape)
    for i in range(len(true)):
        r_true[i] = gaussian_filter(true[i], sigma=1)
    true = map_list_back(r_true)

    return ndcg_score(true, pred, k=k)


def DG_filter(target):
    temp = target.cpu().numpy()
    temp = temp.copy()
    out = np.zeros((64, 80))
    ctr = 0
    for x in range(64):
        for y in range(80):

            if mask[x][y] == 1:
                out[x][y] = temp[ctr]
                ctr += 1

    out = gaussian_filter(out, sigma=0.5)
    temp = np.zeros(len(temp))
    ctr = 0
    for x in range(64):
        for y in range(80):

            if mask[x][y] == 1:
                temp[ctr] = out[x][y]
                ctr += 1

    return torch.tensor(temp)


def loc_ndcg(true, pred, k):
    sum_ndcg = []
    for j in range(true.shape[0]):
        top_k_rank = np.argsort(pred[j])[::-1][:k]
        for key in top_k_rank:
            loc_true = true[j][big_nei_dict_np[key]]
            loc_pred = pred[j][big_nei_dict_np[key]]
            local_ndcg = ndcg_score(np.expand_dims(loc_true, axis=0), np.expand_dims(loc_pred, axis=0))
            sum_ndcg.append(local_ndcg)
    sum_ndcg = np.array(sum_ndcg)

    return np.mean(sum_ndcg)


def transform_top_k(temp, k):
    target = temp.copy()
    for t in range(target.shape[0]):

        top_ranked = np.argsort(target[t])[::-1][:k]

        target[t][:] = 0

        for each in top_ranked:
            target[t][each] = 1

    return target


def map_squre_back(temp):
    target = temp.copy()
    out = np.zeros((target.shape[0], 1, 64, 80))
    for t in range(target.shape[0]):
        ctr = 0
        for x in range(80):
            for y in range(64):

                if mask[x][y] == 1:
                    out[t][0][x][y] = target[t][ctr]
                    ctr += 1
    return out


def map_list_back(temp):
    target = temp.copy()
    out = np.zeros((target.shape[0], int(np.sum(mask))))
    for t in range(target.shape[0]):
        ctr = 0
        for x in range(64):
            for y in range(80):

                if mask[x][y] == 1:
                    out[t][ctr] = target[t][x][y]
                    ctr += 1
    return out


def precision_recall_top_k(pred_list, true_list, k):
    total = np.sum(pred_list) + np.sum(true_list)

    pred_list = pred_list.astype(int)
    true_list = true_list.astype(int)

    if np.sum(pred_list) + np.sum(true_list) != total:
        print("warning!: inputs must be integer to calculate recall and precision")

    new_pred, new_true = [], []

    for t in range(len(pred_list)):
        idx = np.argsort(pred_list[t])[::-1][:k]
        new_pred.append(pred_list[t][idx])
        new_true.append(true_list[t][idx])

    new_pred = np.array(new_pred).flatten()
    new_true = np.array(new_true).flatten()

    precision = accuracy_score(new_pred, new_true)

    return precision, np.sum(new_true) / np.sum(true_list)


def precision_recall_metric_top_k(true, pred, k):
    spatial_ndcg = loc_ndcg(true, pred, k)

    pred = transform_top_k(pred, k)
    true = np.where(true > 0, 1, 0)

    return precision_recall_top_k(pred, true, k)[0], spatial_ndcg

# print(precision_recall_metric_top_k(Yt, temp, k=100))
# print("NDCG: " + str(ndcg_score(Yt, temp, k=100)))

def position_dict_generate(mask):

    ctr = 0
    map_to_list = {}
    list_to_map = {}

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):

            if mask[x][y] == 1:
                map_to_list[str(x) + "-" + str(y)] = ctr
                list_to_map[ctr] = [x, y]
                ctr += 1

    return map_to_list, list_to_map


def neibour_dict(mask, distance):
    nei_dict = {}

    len_x, len_y = mask.shape

    for x1 in range(len_x):
        for y1 in range(len_y):

            if mask[x1][y1] != 1:
                continue

            nei_dict[map_to_list[str(x1) + "-" + str(y1)]] = {}

            for x2 in range(len_x):
                for y2 in range(len_y):

                    if mask[x2][y2] != 1:
                        continue

                    dist = abs(x1 - x2) + abs(y1 - y2)  # manhattan distance

                    if dist <= distance:
                        val = 1
                    else:
                        val = 0

                    nei_dict[map_to_list[str(x1) + "-" + str(y1)]][map_to_list[str(x2) + "-" + str(y2)]] = val

    return nei_dict


def neibour_dictionary_generator(mask, distance):
    nei_dict = {}

    len_x, len_y = mask.shape

    for x1 in range(len_x):
        for y1 in range(len_y):

            if mask[x1][y1] != 1:
                continue

            nei_dict[map_to_list[str(x1) + "-" + str(y1)]] = []

            for x2 in range(len_x):
                for y2 in range(len_y):

                    if mask[x2][y2] != 1:
                        continue

                    if abs(x1 - x2) <= distance and abs(y1 - y2) <= distance:  # window distance
                        nei_dict[map_to_list[str(x1) + "-" + str(y1)]].append(map_to_list[str(x2) + "-" + str(y2)])

    return nei_dict

def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def map_squre_back_for_X(target):
    out = torch.zeros(target.shape[0], 7, 80, 64, target.shape[3]) * 1.0
    for t in range(target.shape[0]):
        for seq in range(7):
            ctr = 0
            for x in range(80):
                for y in range(64):

                    if mask[x][y] == 1:
                        out[t][seq][x][y] = target[t][seq][ctr]
                        ctr += 1
    return out


def map_list_back_for_y(target):
    new_X = []
    for i in range(80):
        for j in range(64):
            if mask[i][j] == 1:
                new_X.append(target[:, i, j])
    new_X = torch.stack(new_X)
    new_X = torch.swapaxes(new_X, 0, 1)

    return new_X
