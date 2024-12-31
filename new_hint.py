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

# the part that dynamic adjancy matrix generated is based nodes' information and it is used as a part of building block of our model. 
# generated dynamic adjancy is implemented based on this paper "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural 

class SpatialRank(nn.Module):
    def __init__(self, num_temporal=None, num_spatial=None, num_spatial_tempoal=None, map_size=None, input_len=None):
        super(SpatialRank, self).__init__()
        self.num_temporal = num_temporal
        self.num_spatial = num_spatial
        self.num_spatial_tempoal = num_spatial_tempoal
        self.map_size = map_size
        self.input_len = input_len
        self.muti1 = 32
        self.muti2 = 4
        self.alpha = 1

        cnn_list = []

        for i in range(0, 7):
            cnn_list.append(GraphConv(num_spatial_tempoal, num_spatial_tempoal))

        self.cnn3d = nn.ModuleList(cnn_list)

        # self.bn1 = nn.BatchNorm2d(15)
        self.two_d_cnn = GraphConv(num_spatial, num_spatial)

        self.LSTM = nn.LSTM((num_temporal + num_spatial_tempoal * self.muti1),
                            hidden_size=num_temporal + num_spatial_tempoal * self.muti1,
                            batch_first=True)

        self.LSTM_sp = nn.LSTM((num_spatial_tempoal*self.map_size),
                            hidden_size=num_spatial_tempoal*self.map_size,
                            batch_first=True)

        self.LSTM_t = nn.LSTM((num_temporal),
                            hidden_size=num_temporal,
                            batch_first=True)

        self.FC = nn.Linear(map_size * num_spatial_tempoal * input_len, num_spatial_tempoal * input_len * self.muti1)
        self.FC2 = nn.Linear(map_size * num_spatial, num_spatial * self.muti2)
        self.FC3 = nn.Linear(num_temporal + num_spatial * self.muti2 + num_spatial_tempoal * self.muti1, map_size)
        self.FC_t = nn.Linear(num_temporal, 1)
        self.FC_sp = nn.Linear(num_spatial_tempoal, num_spatial_tempoal)
        self.FC_sp2 = nn.Linear(num_spatial_tempoal, num_spatial_tempoal)
        self.prelu = nn.ReLU()

    def forward(self, x, A_s):
        b = x.shape[0]
        temporal_view = x[:, :, 0, 0:self.num_temporal].clone()
        spatial_view = x[:, 0, :, self.num_temporal:self.num_temporal + self.num_spatial].clone()
        spatial_tempoal_view = x[:, :, :,
                               self.num_temporal + self.num_spatial:self.num_temporal + self.num_spatial + self.num_spatial_tempoal].clone()

        current, (h_n, c_n) = self.LSTM_sp(spatial_tempoal_view.flatten(2).clone())
        h_n = torch.reshape(torch.squeeze(h_n), (b, self.map_size, self.num_spatial_tempoal))
        Z_s = torch.tanh(self.alpha*self.FC_sp(h_n))
        Z_t = torch.tanh(self.alpha*self.FC_sp2(h_n))
        A_d = self.prelu(torch.tanh(torch.bmm(Z_s, torch.transpose(Z_t, 1, 2)) - torch.bmm(Z_t, torch.transpose(Z_s, 1, 2))))
        current, (h_n, c_n) = self.LSTM_t(temporal_view)
        h_n = torch.squeeze(h_n)
        h_n = self.FC_t(h_n)
        alpha = torch.sigmoid(h_n)

        adj = torch.multiply(torch.unsqueeze(alpha,1).expand(-1, self.map_size, self.map_size), A_d) - torch.multiply(torch.unsqueeze(1-alpha,1).expand(-1, self.map_size, self.map_size), A_s)

        for i in range(7):
            spatial_tempoal_view[:, i] = self.cnn3d[i](spatial_tempoal_view[:, i].clone(), adj)

        spatial_view = self.two_d_cnn(spatial_view, adj)

        spatial_tempoal_view = self.FC(spatial_tempoal_view.flatten(1))
        spatial_tempoal_view = torch.reshape(spatial_tempoal_view,
                                             (len(x), self.input_len, self.num_spatial_tempoal * self.muti1))

        merged_two_view = torch.cat((spatial_tempoal_view, temporal_view), 2)

        current, (h_n, c_n) = self.LSTM(merged_two_view)

        merged_two_view = h_n.permute(1, 0, 2).flatten(1)
        spatial_view = self.FC2(spatial_view.flatten(1))

        final_view = torch.cat((merged_two_view, spatial_view), 1)
        final_view = self.FC3(final_view)

        return final_view