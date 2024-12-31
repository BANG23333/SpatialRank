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
from GC import GraphConv
from new_hint import SpatialRank
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

K = 30
alhpa_real = 0.1 # hyper

# load dataset

# the orginal study area is parititioned into a grid as a rectagle with wide len_x, len_y.
# The shape of the X would be like [samples, sequence_length, len_x, len_y, features]
# To formulate a ranking problem, we reformulate current X and Y to a new list of locations by a mask map

# mask [len_x, len_y] is the map of study area(rectangle shape), where mask[x][y] = 1 means this location is considered in the ranking, 0 # # means it is ignored.

# By using mask map, we only consider some locations and reshape the X and Y into:

# The shape of X: [samples, sequence_length, length_of_list(number of locations), features]
# The shape of Y: [samples, length_of_list(number of locations)]

# the below X, Y, Xv, Yv, Xt, Yt are reshaped matrix (list of locations)

# hyperoparamter num_temporal, num_spatial, num_spatial_tempoal, are the numebr of temporal feature diemnsions, spatial feature dims, and # spatialtemporal feature dims in your X[:, :, :, !] (feature dim)



X = np.load(open('X.npy', 'rb'))
Y = np.load(open('Y.npy', 'rb'))

Xv = np.load(open('Xv.npy', 'rb'))
Yv = np.load(open('Yv.npy', 'rb'))

Xt = np.load(open('Xt.npy', 'rb'))
Yt = np.load(open('Yt.npy', 'rb'))

X_index = np.arange(len(X))
Xv_index = np.arange(len(X)+len(Xv))
Xt_index = np.arange(len(X)+len(Xv)+len(Xt))

x_len, y_len = X.shape[1], X.shape[2]
print(x_len, y_len)

num_temporal=12
num_spatial=18
num_spatial_tempoal=6
map_size=Y.shape[1]
input_len=7

mask = np.load(open('mask.npy', 'rb')) # mask[x][y] = 1 if this cell is considered in the ranking list.

from torch.utils.data import Dataset

class Test_Dataset(Dataset):
    def __init__(self, X_input, X_index, Y_input):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.X_index = Variable(torch.Tensor(X_index).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.X_index[idx], self.Y_input[idx]

def normalize(X):

    x_max = np.zeros(X.shape[3])
    x_min = np.zeros(X.shape[3])
    x_dif = np.zeros(X.shape[3])
    
    for i in range(X.shape[3]):
        x_max[i] = np.max(X[:,:,:,i])
        x_min[i] = np.min(X[:,:,:,i])

    x_dif = x_max - x_min
    
    x_dif = np.where(x_dif==0, 1, x_dif)
                
    new_X = np.zeros(X.shape)
        
    for ctr in range(X.shape[0]):
        for x in range(X.shape[1]):
            for y in range(X.shape[2]):
                new_X[ctr][x][y] = (X[ctr][x][y] - x_min)/x_dif
                    
    return new_X

class RandomSampledData(Dataset):
    def __init__(self, X_input, Y_input, num_pos, num_all):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())
        self.num_pos, self.num_all = num_pos, num_all

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        pos_idx = (self.Y_input[idx]> 0).nonzero().squeeze()
        neg_idx = (self.Y_input[idx]==0).nonzero().squeeze()
        
        filtered_Y = np.expand_dims(self.Y_input[idx].numpy(), axis=0)
        filtered_Y = np.squeeze(map_squre_back(filtered_Y))
        filtered_Y = gaussian_filter(filtered_Y, sigma=1)
        filtered_Y = np.expand_dims(filtered_Y, axis=0)
        filtered_Y = map_list_back(filtered_Y)
        filtered_Y = np.squeeze(filtered_Y)
        
        weights_pos = dynamic_DG[(self.Y_input[idx]> 0).nonzero().squeeze()]
        weights_neg = torch.ones(len(self.Y_input[idx][(self.Y_input[idx]== 0).nonzero().squeeze()]))
        
        perm_pos = torch.multinomial(weights_pos, self.num_pos, replacement=True)
        perm_neg = torch.multinomial(weights_neg, self.num_all-self.num_pos, replacement=True)

        sampled_pos_idx = pos_idx[perm_pos]
        sampled_neg_idx = neg_idx[perm_neg]
        
        sampled_idx = torch.concat([sampled_pos_idx, sampled_neg_idx])

        label = torch.tensor([1]*len(sampled_pos_idx) + [0]*len(sampled_neg_idx))
        
        loc_y, loc_index = [], []
        for each in sampled_idx.cpu().detach().numpy():
            loc_index.append(big_nei_dict[each])
            loc_y.append(self.Y_input[idx][big_nei_dict[each]])
        
        loc_index = torch.stack(loc_index)
        loc_y = torch.stack(loc_y)

        return self.X_input[idx], self.Y_input[idx], idx, self.Y_input[idx][sampled_idx], label, sampled_idx, filtered_Y[sampled_idx], loc_index, loc_y

def Discounted_Gain(y, ranking):
    
    nomi = 2**y - 1
    denomi = np.log2(2 + ranking)
    return nomi/denomi
    

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

    return out/dif.shape[0]

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
    out = np.zeros((x_len, y_len))
    ctr = 0
    for x in range(x_len):
        for y in range(y_len):

            if mask[x][y] == 1:
                out[x][y] = temp[ctr]
                ctr += 1
                
    out = gaussian_filter(out, sigma=0.5)
    temp = np.zeros(len(temp))
    ctr = 0
    for x in range(x_len):
        for y in range(y_len):

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
    out = np.zeros((target.shape[0], 1, x_len, y_len))
    for t in range(target.shape[0]):
        ctr = 0
        for x in range(x_len):
            for y in range(y_len):
    
                if mask[x][y] == 1:
                    out[t][0][x][y] = target[t][ctr]
                    ctr += 1
    return out

def map_list_back(temp):
    
    target = temp.copy()
    out = np.zeros((target.shape[0], int(np.sum(mask))))
    for t in range(target.shape[0]):
        ctr = 0
        for x in range(x_len):
            for y in range(y_len):
    
                if mask[x][y] == 1:
                    out[t][ctr] = target[t][x][y]
                    ctr += 1
    return out
    

def precision_recall_top_k(pred_list, true_list, k):
    
    total = np.sum(pred_list)+np.sum(true_list)
    
    pred_list = pred_list.astype(int)
    true_list = true_list.astype(int)
    
    if np.sum(pred_list)+np.sum(true_list) != total:
        print("warning!: inputs must be integer to calculate recall and precision")
    
    new_pred, new_true = [], []
        
    for t in range(len(pred_list)):
        
        idx = np.argsort(pred_list[t])[::-1][:k]
        new_pred.append(pred_list[t][idx])
        new_true.append(true_list[t][idx])
        
    new_pred = np.array(new_pred).flatten()
    new_true = np.array(new_true).flatten()
    
    precision = accuracy_score(new_pred, new_true)

    return precision, np.sum(new_true)/np.sum(true_list)

def precision_recall_metric_top_k(true, pred, k):
    
    # -------------------------------------------------------
    spatial_ndcg = loc_ndcg(true, pred, k)

    pred = transform_top_k(pred, k)
    true = np.where(true>0, 1, 0)
    
    return precision_recall_top_k(pred, true, k)[0], spatial_ndcg

try:
    with open('p_corr_chicago_64_80.pickle', 'rb') as handle:
        p_corr_dict = pickle.load(handle)
except:
    # if PS coef is not available
    p_corr_dict = {}
    for i in range(Y.shape[1]):
        p_corr_dict[i] = {}
        for j in range(Y.shape[1]):
            p_corr_dict[i][j] = 0.0
            
def position_dict_generate(mask):
    
    ctr = 0
    map_to_list = {}
    list_to_map = {}
    
    
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            
            if mask[x][y] == 1:
                
                map_to_list[str(x)+"-"+str(y)] = ctr
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
    
            nei_dict[map_to_list[str(x1)+"-"+str(y1)]] = {}
    
            for x2 in range(len_x):
                for y2 in range(len_y):
                
                    if mask[x2][y2] != 1:
                        continue    
                        
                    dist =  abs(x1-x2)+abs(y1-y2) # manhattan distance
                    
                    if dist <= distance:
                        val = 1
                    else:
                        val = 0
                        
                    nei_dict[map_to_list[str(x1)+"-"+str(y1)]][map_to_list[str(x2)+"-"+str(y2)]] = val
    
    return nei_dict

def generate_adj(sampled_idx, device):
    
    sample_num = len(sampled_idx)
    adj = torch.zeros(sample_num, sample_num, device=device)
    # sampled_idx = sampled_idx.numpy()
    
    for i in range(sample_num):
        for j in range(sample_num):
            
            if nei_dict[sampled_idx[i]][sampled_idx[j]] == 1:
                adj[i][j] = 1
            else:
                adj[i][j] = p_corr_dict[sampled_idx[i]][sampled_idx[j]]
            
    return adj


map_to_list, list_ot_map = position_dict_generate(mask)
nei_dict = neibour_dict(mask, 2)

def neibour_dictionary_generator(mask, distance):

    nei_dict = {}
    
    len_x, len_y = mask.shape
    
    for x1 in range(len_x):
        for y1 in range(len_y):
            
            if mask[x1][y1] != 1:
                continue
    
            nei_dict[map_to_list[str(x1)+"-"+str(y1)]] = []
    
            for x2 in range(len_x):
                for y2 in range(len_y):
                
                    if mask[x2][y2] != 1:
                        continue    
                                            
                    if abs(x1-x2) <= distance and abs(y1-y2) <= distance: # window distance
                        nei_dict[map_to_list[str(x1)+"-"+str(y1)]].append(map_to_list[str(x2)+"-"+str(y2)])
                        
    return nei_dict

radius = 2
big_nei_dict_np = neibour_dictionary_generator(mask, radius)
big_nei_dict = {}

for key in big_nei_dict_np.keys():
    while len(big_nei_dict_np[key]) < (radius*2 + 1)**2:
        big_nei_dict_np[key].append(key)
    
for key in big_nei_dict_np.keys():
    big_nei_dict[key] = torch.tensor(big_nei_dict_np[key])


import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import libauc
from libauc.datasets.movielens import *
from libauc.sampler.ranking import DataSampler
from libauc.losses.ranking import NDCG_Loss, ListwiseCE_Loss
from libauc.optimizers import SONG
# from libauc.models import NeuMF
from libauc.utils.helper import batch_to_gpu, adjust_lr, format_metric, get_time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ndcg_score
from sklearn.metrics import accuracy_score

class MoiveLens(Dataset):

      def __init__(self, root, phase='train', topk=-1, random_seed=123, n_users=None, n_items=None):
       
          self.n_users = n_users           # number of users in the dataset
          self.n_items = n_items           # number of items in the dataset         
                          
          if phase == 'train':
             dataset = moivelens_trainset(data_path=os.path.join(root, 'train.csv'), n_users=self.n_users, n_items=self.n_items, topk=topk)
             self.id_mapper = dataset.get_id_mapper()
          else:                  
             dataset = moivelens_evalset(data_path=os.path.join(root, phase+'.csv'), n_users=self.n_users, n_items=self.n_items, phase=phase) 

          self.dataset = dataset
          self.phase = phase
          self.targets = dataset.targets

      def get_id_mapper(self):
          assert self.phase == 'train'  
          return self.id_mapper

      def collate_batch(self, feed_dicts):
          return self.dataset.collate_batch(feed_dicts)

      def get_batch(self, index: int, batchsize: int):
          assert self.phase in ['dev', 'test']
          return self.dataset.get_batch(index, batchsize)        
        
      def __len__(self):
          return self.dataset.__len__()

      def __getitem__(self, index):
          assert self.phase == 'train'
          return self.dataset.__getitem__(index)
    
class Test_Dataset(Dataset):
    def __init__(self, X_input, X_index, Y_input):
        self.X_input = Variable(torch.Tensor(X_input).float())
        self.X_index = Variable(torch.Tensor(X_index).float())
        self.Y_input = Variable(torch.Tensor(Y_input).float())

    def __len__(self):
        return len(self.X_input)

    def __getitem__(self, idx):
        return self.X_input[idx], self.X_index[idx], self.Y_input[idx]

DATA_PATH = 'index'

LOSS = 'SONG'             
NUM_POS = 256              
NUM_NEG = 256             
L2 = 1e-7                 
OPTIMIZER_STYLE = 'adam'  
n_users = 790
n_items = Y.shape[1]

trainSet = MoiveLens(root=DATA_PATH, phase='train', n_users=n_users, n_items=n_items)

GAMMA0 = 0.1

id_mapper, num_relevant_pairs = trainSet.get_id_mapper()

class NDCG_Neighbor_Loss(nn.Module):
    def __init__(self, user_num: int, item_num: int, num_pos: int,
                 gamma0: float, gamma1: float=0.9, eta0: float=0.01,
                 sqh_c: float=1.0, k: int=-1, topk_version: str='theo', tau_1: float=0.001, tau_2: float=0.0001,
                 psi_func: str='sigmoid', hinge_margin: float=1.0, c_sigmoid: float=2.0, sigmoid_alpha: float=2.0):
        super(NDCG_Neighbor_Loss, self).__init__()
        self.num_pos = num_pos
        self.sqh_c = sqh_c
        self.gamma0 = gamma0
        self.k = k                               
        self.lambda_q = torch.zeros(user_num+1)  # learnable thresholds for all querys (users)
        self.v_q = torch.zeros(user_num+1)       # moving average estimator for \nabla_{\lambda} L_q
        self.gamma1 = gamma1                        
        self.tau_1 = tau_1                            
        self.tau_2 = tau_2                       
        self.eta0 = eta0                  
        self.item_num = item_num
        self.topk_version = topk_version         # theo: sigmoid_alpha=2.0 ; prac: sigmoid_alpha=0.01
        self.s_q = torch.zeros(user_num+1)       # moving average estimator for \nabla_{\lambda}^2 L_q
        self.psi_func = psi_func
        self.hinge_margin = hinge_margin
        self.sigmoid_alpha = sigmoid_alpha
        self.c_sigmoid = c_sigmoid
        self.u = torch.zeros(user_num+1, item_num+1)
            
    def _squared_hinge_loss(self, x, c):
        return torch.max(torch.zeros_like(x), x + c) ** 2

    def forward(self, loc_predictions: torch.Tensor, batch) -> torch.Tensor:


        loss, ctr = 0, 0
        
        for i in range(self.item_num):
            
            self.num_pos = batch["loc_pos"][0, i]
            predictions = loc_predictions[:, i]
            device = predictions.device
            ratings = batch['rating'][:, i][:, :self.num_pos]                                                           # [batch_size, num_pos]
            batch_size = ratings.size()[0]
            predictions_expand = einops.repeat(predictions, 'b n -> (b copy) n', copy=self.num_pos)  # [batch_size*num_pos, num_pos+num_neg]
            predictions_pos = einops.rearrange(predictions[:, :self.num_pos], 'b n -> (b n) 1')      # [batch_suze*num_pos, 1]

            num_pos_items = batch['num_pos_items'].float()  # [batch_size], the number of positive items for each user
            ideal_dcg = batch['ideal_dcg'][:, i].float()          # [batch_size], the ideal dcg for each user
            g = torch.mean(self._squared_hinge_loss(predictions_expand-predictions_pos, self.sqh_c), dim=-1)   # [batch_size*num_pos]
            g = g.reshape(batch_size, self.num_pos)                                                            # [batch_size, num_pos], line 5 in Algo 2.

            G = (2.0 ** ratings - 1).float()

            user_ids = batch['user_id']
            pos_item_ids = batch['item_id'][:, i][:, :self.num_pos]  # [batch_size, num_pos]

            pos_item_ids = einops.rearrange(pos_item_ids, 'b n -> (b n)')
            user_ids_repeat = einops.repeat(user_ids, 'n -> (n copy)', copy=self.num_pos)

            self.u[user_ids_repeat, pos_item_ids] = (1-self.gamma0) * self.u[user_ids_repeat, pos_item_ids] + self.gamma0 * g.clone().detach_().reshape(-1).cpu()
            g_u = self.u[user_ids_repeat, pos_item_ids].reshape(batch_size, self.num_pos).to(device)

            nabla_f_g = (G * self.item_num) / ((torch.log2(1 + self.item_num*g_u))**2 * (1 + self.item_num*g_u) * np.log(2)) # \nabla f(g)

            if self.k > 0:
                pos_preds_lambda_diffs = predictions[:, :self.num_pos].clone().detach_() - self.lambda_q[user_ids][:, None].to(device)
                preds_lambda_diffs = predictions.clone().detach_() - self.lambda_q[user_ids][:, None].to(device)

                # the gradient of lambda
                grad_lambda_q = self.k/self.item_num + self.tau_2*self.lambda_q[user_ids] - torch.mean(torch.sigmoid(preds_lambda_diffs.cpu() / self.tau_1), dim=-1)
                self.v_q[user_ids] = self.gamma1 * grad_lambda_q + (1-self.gamma1) * self.v_q[user_ids]
                self.lambda_q[user_ids] = self.lambda_q[user_ids] - self.eta0 * self.v_q[user_ids]

                if self.topk_version == 'prac':
                    if self.psi_func == 'hinge':
                        nabla_f_g *= torch.max(pos_preds_lambda_diffs+self.hinge_margin, torch.zeros_like(pos_preds_lambda_diffs))
                    elif self.psi_func == 'sigmoid':
                        nabla_f_g *= self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                    else:
                        assert 0, "psi_func " + self.psi_func + " is not supported."

                elif self.topk_version == 'theo':
                    if self.psi_func == 'hinge':
                        nabla_f_g *= torch.max(pos_preds_lambda_diffs+self.hinge_margin, torch.zeros_like(pos_preds_lambda_diffs))
                        weight_1 = (pos_preds_lambda_diffs+self.hinge_margin > 0).float()
                    elif self.psi_func == 'sigmoid':
                        nabla_f_g *= self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                        weight_1 = self.c_sigmoid * torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha) * (1 - torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha))
                    else:
                        assert 0, "psi_func " + self.psi_func + " is not supported."

                    temp_term = torch.sigmoid(preds_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(preds_lambda_diffs / self.tau_1)) / self.tau_1
                    L_lambda_hessian = self.tau_2 + torch.mean(temp_term, dim=1)
                    self.s_q[user_ids] = self.gamma1 * L_lambda_hessian.cpu() + (1-self.gamma1) * self.s_q[user_ids]
                    hessian_term = torch.mean(temp_term * predictions, dim=1) / self.s_q[user_ids].to(device)
                    f_g_u = -G / torch.log2(1 + self.item_num*g_u)
                    loss = (num_pos_items * torch.mean(nabla_f_g * g + weight_1 * f_g_u * (predictions[:, :self.num_pos] - hessian_term[:, None]), dim=-1) / ideal_dcg).mean()
                    return loss
            
            tmp_loss = (num_pos_items * torch.mean(nabla_f_g * g, dim=-1) / ideal_dcg).mean()
            if not torch.isnan(tmp_loss):
                loss += tmp_loss
                ctr += 1
            
        return loss/ctr
    

class Hybrid_Loss(nn.Module):
    def __init__(self, alhpa=0.1):
        super(Hybrid_Loss, self).__init__()
        self.alhpa = alhpa
        self.ndcg_loss = NDCG_Loss(id_mapper, num_relevant_pairs, n_users, n_items, NUM_POS, gamma0=GAMMA0, topk=-1, topk_version='prac')
        self.ndcg_loss2 = NDCG_Neighbor_Loss(n_users, 512, NUM_POS, k=-1, gamma0=0.2, topk_version="prac")

    def forward(self, outputs, label_index, local_labels, local_index, filtered_Y, loc_index, loc_y):
        outputs2 = outputs.clone()
        label_index2 = label_index.clone()
        local_index2 = local_index.clone()
        
        num_labels = torch.count_nonzero(outputs)
        tmp = []
        idcg = []
        for i,j,k in zip(outputs, label_index, local_labels):
            
            tmp.append(i[j].view(1,-1))
            k = k.cpu().numpy()
            idcg.append(dcg_score(k,k,K))
            outputs = torch.concat(tmp)
        
        idcg = torch.tensor(idcg).to(device)
        batch = {'user_id': local_index, 'item_id': label_index, 'rating':local_labels , 'num_pos_items': num_labels, 'ideal_dcg': idcg}
        g_loss = self.ndcg_loss(outputs, batch)
        
        idcg = []
        loc_output = []
        for i in range(loc_index.shape[0]):
            loc_output_tmp = []
            for j in range(loc_index.shape[1]):
                loc_output_tmp.append(outputs2[i][loc_index[i][j]])
            loc_output_tmp = torch.stack(loc_output_tmp)
            loc_output.append(loc_output_tmp)
        loc_output = torch.stack(loc_output)
        
        num_labels = torch.count_nonzero(outputs2)
        loc_pos = torch.count_nonzero(loc_y, dim=2)
        
        for i in range(loc_index.shape[1]):
        
            tmp_idcg = []
            for k in loc_y[:, i]:
                k = k.cpu().detach().numpy()
                tmp_idcg.append(dcg_score(k,k,K))
            
            idcg.append(tmp_idcg)
        
        idcg = np.array(idcg)
        idcg = np.einsum('ab->ba', idcg)
        
        idcg = torch.tensor(idcg).to(device)
        batch = {'user_id': local_index2, 'item_id': loc_index, 'rating':loc_y , 'num_pos_items': num_labels, 'ideal_dcg': idcg, 'loc_pos': loc_pos}
        local_loss = self.ndcg_loss2(loc_output, batch)
        
        loss = (1-self.alhpa)*g_loss + self.alhpa*local_loss
        
        return loss
    

def dcg_score(y_true, y_score, k=10, gains="exponential"):

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
    out = torch.zeros(target.shape[0], 7, x_len, y_len, target.shape[3])*1.0
    for t in range(target.shape[0]):
        for seq in range(7):
            ctr = 0
            for x in range(x_len):
                for y in range(y_len):

                    if mask[x][y] == 1:
                        out[t][seq][x][y] = target[t][seq][ctr]
                        ctr += 1
    return out

def map_list_back_for_y(target):
        
    new_X = []
    for i in range(x_len):
        for j in range(y_len):
            if mask[i][j] == 1:
                new_X.append(target[:, i, j])
    new_X = torch.stack(new_X)
    new_X = torch.swapaxes(new_X, 0, 1)
    
    return new_X


batch_size = 64
criterion = Hybrid_Loss(alhpa=alhpa_real).to(device)

model = SpatialRank(num_temporal=num_temporal, num_spatial=num_spatial, num_spatial_tempoal=num_spatial_tempoal, map_size=Y.shape[1], input_len=input_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

checkpoint = torch.load('check_point_hint_CE.pth') # warm up model
model.load_state_dict(checkpoint['model_state_dict'])

for g in optimizer.param_groups:
    g['lr'] = 1e-4
    
sampled_idx = np.arange(Y.shape[1])
local_adj = generate_adj(sampled_idx, device)
print(local_adj.get_device())

train_loss_arr = []
valid_loss_arr =[]

train_ndcg_arr = []
valid_ndcg_arr =[]
valid_prec_arr =[]
valid_recall_arr =[]

validation_dataset = Test_Dataset(Xv, Xv_index, Yv)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

train_dataset = RandomSampledData(X,Y,NUM_POS,Y.shape[1])
training_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

best_ndcg = 0
    
dynamic_DG = torch.ones(Y.shape[1])

for echo in range(80):        
    
    print("echo: " + str(echo))
    avg_train_loss = []
    avg_valid_loss = []
    avg_train_ndcg = []
    avg_valid_ndcg = []
    avg_valid_prec = []
    avg_valid_recall = []

    # model.train()
    true_labels = []
    predictions = []
    # print("=============================")
    
    for local_batch, true_la, local_index, local_labels, pos_index, label_index, filtered_Y, loc_index, loc_y in training_generator:
        local_batch, local_index, local_labels, label_index, filtered_Y = local_batch.to(device), local_index.to(device), local_labels.to(device), label_index.to(device), filtered_Y.to(device)
        true_la = true_la.to(device)
        loc_index, loc_y = loc_index.to(device), loc_y.to(device)

        pos_index = pos_index.to(device)
        outputs = model(local_batch, local_adj)
        outputs = torch.squeeze(outputs)

        train_loss = criterion(outputs, label_index, local_labels, local_index, filtered_Y, loc_index, loc_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
#         outputs = outputs[label_index].cpu().detach().numpy()
        avg_train_loss.append(train_loss.cpu().data)
        avg_train_ndcg.append(ndcg_score(local_labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), k=K))

        true_labels.append(true_la)
        predictions.append(outputs)
            
    true_labels = torch.cat(true_labels)
    predictions = torch.cat(predictions)
    diff = torch.abs(true_labels - predictions)
    
    DG_list = DG_mapping(diff.cpu().detach().numpy(), true_labels.cpu().detach().numpy())
    
    to_plot = np.zeros((x_len, y_len))
    
    dynamic_DG = torch.tensor(DG_list)
    dynamic_DG = DG_filter(dynamic_DG)
    
    train_ndcg_arr.append(np.average(avg_train_ndcg))
    
    with torch.no_grad():
        for local_batch, local_index, local_labels in validation_generator:
            local_batch, local_index, local_labels = local_batch.to(device), local_index.to(device), local_labels.to(device)
            
            Voutputs = model(local_batch, local_adj)
            Voutputs = torch.squeeze(Voutputs)
            
            avg_valid_ndcg.append(ndcg_score(local_labels.cpu().detach().numpy(), Voutputs.cpu().detach().numpy(), k=K))
            prec, recall = precision_recall_metric_top_k(local_labels.cpu().detach().numpy(), Voutputs.cpu().detach().numpy(), k=K)
            avg_valid_prec.append(prec)
            avg_valid_recall.append(recall)
            
    valid_ndcg_arr.append(np.average(avg_valid_ndcg))
    valid_prec_arr.append(np.average(avg_valid_prec))
    valid_recall_arr.append(np.average(avg_valid_recall))
        
    if best_ndcg < np.average(avg_valid_ndcg):
        torch.save(model,'best_model.pth') 
        print("save the best model")
        best_ndcg = np.average(avg_valid_ndcg)    
        
        early_ctr = 0
    else:
        early_ctr += 1
        
    if early_ctr >= 10:
        # break
        print("  ")
        
    print("----------------------------------------------------------")
    


# In[ ]:


model = torch.load('best_model.pth.pth').to(device)

test_dataset = Test_Dataset(Xt, Xt_index, Yt)
test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

avg_train_loss = []
avg_valid_loss = []
avg_train_ndcg = []
avg_valid_ndcg = []
avg_valid_prec = []
avg_valid_recall = []

model.eval()

with torch.no_grad():
    for local_batch, local_index, local_labels in test_generator:
        local_batch, local_index, local_labels = local_batch.to(device), local_index.to(device), local_labels.to(device)

        Voutputs = model(local_batch, local_adj)
        Voutputs = torch.squeeze(Voutputs)

        avg_valid_ndcg.append(ndcg_score(local_labels.cpu().detach().numpy(), Voutputs.cpu().detach().numpy(), k=K))
        prec, recall = precision_recall_metric_top_k(local_labels.cpu().detach().numpy(), Voutputs.cpu().detach().numpy(), k=K)
        avg_valid_prec.append(prec)
        avg_valid_recall.append(recall)
plot_map_true_pred(local_labels.cpu().detach().numpy(), Voutputs.cpu().detach().numpy(), K)

#print("valid loss: " + str(np.average(avg_valid_loss)))
print("valid ndcg: " + str(np.average(avg_valid_ndcg)))
print("valid prec: " + str(np.average(avg_valid_prec)))
print("valid recall: " + str(np.average(avg_valid_recall)))

