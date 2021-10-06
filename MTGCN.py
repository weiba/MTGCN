import numpy as np
import pandas as pd
import time
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops

from sklearn import metrics

EPOCH = 2500

data = torch.load("./data/CPDB_data.pkl")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
y_all = np.logical_or(data.y, data.y_te)
mask_all = np.logical_or(data.mask, data.mask_te)
data.x = data.x[:, :48]

datas = torch.load("./data/str_fearures.pkl")
data.x = torch.cat((data.x, datas), 1)
data = data.to(device)

with open("./data/k_sets.pkl", 'rb') as handle:
    k_sets = pickle.load(handle)

pb, _ = remove_self_loops(data.edge_index)
pb, _ = add_self_loops(pb)
E = data.edge_index


def train(mask):
    model.train()
    optimizer.zero_grad()

    pred, rl, c1, c2 = model()

    loss = F.binary_cross_entropy_with_logits(pred[mask], Y[mask]) / (c1 * c1) + rl / (c2 * c2) + 2 * torch.log(c2 * c1)
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(mask):
    model.eval()
    x, _, _, _ = model()

    pred = torch.sigmoid(x[mask]).cpu().detach().numpy()
    Yn = Y[mask].cpu().numpy()
    precision, recall, _thresholds = metrics.precision_recall_curve(Yn, pred)
    area = metrics.auc(recall, precision)

    return metrics.roc_auc_score(Yn, pred), area


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(64, 300, K=2, normalization="sym")
        self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
        self.conv3 = ChebConv(100, 1, K=2, normalization="sym")

        self.lin1 = Linear(64, 100)
        self.lin2 = Linear(64, 100)

        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self):
        edge_index, _ = dropout_adj(data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.x.size()[0],
                                    training=self.training)

        x0 = F.dropout(data.x, training=self.training)
        x = torch.relu(self.conv1(x0, edge_index))
        x = F.dropout(x, training=self.training)
        x1 = torch.relu(self.conv2(x, edge_index))

        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()

        neg_edge_index = negative_sampling(pb, 13627, 504378)

        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()

        r_loss = pos_loss + neg_loss


        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return x, r_loss, self.c1, self.c2



time_start = time.time()
#ten five-fold cross-validations
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))

for i in range(10):
    print(i)
    for cv_run in range(5):
        _, _, tr_mask, te_mask = k_sets[i][cv_run]
        model = Net().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.005)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, EPOCH):
            train(tr_mask)

        AUC[i][cv_run], AUPR[i][cv_run] = test(te_mask)


    print(time.time() - time_start)


print(AUC.mean())
print(AUC.var())
print(AUPR.mean())
print(AUPR.var())


model= Net().to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1,EPOCH):
    print(epoch )
    train(mask_all)


x,_,_,_= model()
pred = torch.sigmoid(x[~mask_all]).cpu().detach().numpy()
torch.save(pred, 'pred.pkl')