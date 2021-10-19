import numpy as np
import pandas as pd
import time
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops,add_self_loops



data = torch.load("./data/CPDB_data.pkl")

model =  Node2Vec(data.edge_index, embedding_dim=16, walk_length=80,
                     context_size=5,  walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(1, 501):
    loss = train()
    print (loss)

model.eval()
str_fearures = model()

torch.save(str_fearures, 'str_fearures.pkl')