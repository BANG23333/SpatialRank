# SpatialRank
SpatialRank: Urban Event Ranking with NDCG Optimization on Spatiotemporal Data
Proceedings of the 2023 37th Conference on Neural Information Processing Systems (NeurIPS 2023)

# Abstract
The problem of urban event ranking aims at predicting the top-k most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named SpatialRank. SpatialRank features adaptive graph convolution layers that dynamically learn the spatiotemporal dependencies across locations from data. In addition, the model optimizes through surrogates a hybrid NDCG loss with a spatial component to better rank neighboring spatial locations. We design an importance-sampling with a spatial filtering algorithm to effectively evaluate the loss during training. Comprehensive experiments on three real-world datasets demonstrate that SpatialRank can effectively identify the top riskiest locations of crimes and traffic accidents and outperform state-of-the-art methods in terms of NDCG by up to 12.7%.

![alt text](https://github.com/BANG23333/SpatialRank/blob/main/img/model.jpg)

# Environment
- python 3.7.0
- torch 1.12.1
- matplotlib 3.5.2
- numpy 1.21.5
- sklearn 1.1.1
- seaborn 0.11.2

# Citation
```
@inproceedings{10.5555/3666122.3666555,
author = {An, Bang and Zhou, Xun and Zhong, Yongjian and Yang, Tianbao},
title = {SpatialRank: urban event ranking with NDCG optimization on spatiotemporal data},
year = {2024},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
abstract = {The problem of urban event ranking aims at predicting the top-k most risky locations of future events such as traffic accidents and crimes. This problem is of fundamental importance to public safety and urban administration especially when limited resources are available. The problem is, however, challenging due to complex and dynamic spatio-temporal correlations between locations, uneven distribution of urban events in space, and the difficulty to correctly rank nearby locations with similar features. Prior works on event forecasting mostly aim at accurately predicting the actual risk score or counts of events for all the locations. Rankings obtained as such usually have low quality due to prediction errors. Learning-to-rank methods directly optimize measures such as Normalized Discounted Cumulative Gain (NDCG), but cannot handle the spatiotemporal autocorrelation existing among locations. In this paper, we bridge the gap by proposing a novel spatial event ranking approach named SpatialRank. SpatialRank features adaptive graph convolution layers that dynamically learn the spatiotemporal dependencies across locations from data. In addition, the model optimizes through surrogates a hybrid NDCG loss with a spatial component to better rank neighboring spatial locations. We design an importance-sampling with a spatial filtering algorithm to effectively evaluate the loss during training. Comprehensive experiments on three real-world datasets demonstrate that SpatialRank can effectively identify the top riskiest locations of crimes and traffic accidents and outperform state-of-the-art methods in terms of NDCG by up to 12.7\%.},
booktitle = {Proceedings of the 37th International Conference on Neural Information Processing Systems},
articleno = {433},
numpages = {12},
location = {New Orleans, LA, USA},
series = {NIPS '23}
}
```
