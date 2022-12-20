import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import math

HIDDEN = 1024
LAYER = 7
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5

class GraphChenn(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''
    def __init__(self, in_features, out_features):
        super(GraphChenn, self).__init__()
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = min(1, math.log(lamda/l+1))
        hi = torch.spmm(adj, input)
        support = (1-alpha)*hi+alpha*h0
        r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = output+input
        return output


class GATLayer(nn.Module):
    """
    Changed from GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # W and a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # leakyrelu
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj, h0 , lamda, alpha, l):
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N, num of nodes

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N]

        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        attention = F.softmax(attention, dim=1)  # softmax, [N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        theta = min(1, math.log(lamda / l + 1))
        hi = torch.spmm(attention, inp)
        support = (1 - alpha) * hi + alpha * h0
        r = support
        output = theta * torch.mm(support, self.W) + (1 - theta) * r
        output = output + inp
        return output

class Spatom(nn.Module):
    '''
    changed from https://github.com/chennnM/GCNII
    '''
    def __init__(self, nlayers=LAYER, nfeat=63, nhidden=HIDDEN, dropout=DROPOUT, lamda=LAMBDA, alpha=ALPHA,out_dim =1):
        super(Spatom, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphChenn(nhidden, nhidden))
        self.GAT1 = GATLayer(nhidden, nhidden,dropout,0.2)
        self.fc = nn.Linear(nfeat, nhidden)
        self.classification1 = nn.Linear(nhidden,64)
        #self.classification2 = nn.Linear(512, 512)
        self.classification3 = nn.Linear(64,out_dim)
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, dist , adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fc(x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,dist,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = F.elu(self.GAT1(layer_inner,adj,_layers[0],self.lamda,self.alpha,9))
        layer_inner = self.act_fn(self.classification1(layer_inner))
        #layer_inner = self.act_fn(self.classification2(layer_inner))
        layer_inner = self.sigmoid(self.classification3(layer_inner))
        return layer_inner.squeeze()