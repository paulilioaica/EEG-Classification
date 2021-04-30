import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.gc3 =  GCNConv(nhid, nhid)
        self.linear = nn.Linear(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, batch_size):
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        x = F.relu(self.gc3(x, adj))


        x = global_mean_pool(x, batch_size)  # [batch_size, hidden_channels]
        # x = F.dropout(x, p=0.1, training=self.training)
        x = torch.sigmoid(x)
        x = self.linear(x)
        return x