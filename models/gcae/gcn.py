import torch
import torch.nn as nn
import math



class GraphConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.parameter.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)

        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_channels))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)


    def forward(self, X:torch.Tensor, Adj:torch.Tensor):

        X = torch.einsum('bij,jk->bik', (X,self.weight))
        X = torch.einsum('ij,bjk->bik', (Adj,X))

        if self.bias is not None:
            X = X + self.bias
        
        return X

        

class StaticPlain_GCNN_Layer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 time_dim,
                 joints_dim,
                 bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.joints_dim = joints_dim
        
        self.gcn = GraphConvBlock(in_channels, out_channels, bias)
        self.act = nn.ReLU()


    def forward(self, X:torch.Tensor, Adj:torch.Tensor):

        X = self.act(self.gcn(X, Adj))
        return X

