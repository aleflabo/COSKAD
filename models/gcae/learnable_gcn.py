import torch
import torch.nn as nn
import math



class LearnableGraphConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_frames,
                 n_joints,
                 bias=True
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_frames = n_frames
        self.n_joints = n_joints

        self.weight = nn.parameter.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))

        if bias:
            self.bias = nn.parameter.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.Adj = nn.parameter.Parameter(torch.FloatTensor(n_frames*n_joints,n_frames*n_joints))
        self.adj_act = nn.Softmax()

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.Adj.size(0))
        # self.Adj.data.uniform_(-stdv,stdv)
        self.Adj.data.uniform_(0.0,1.0)
        

    def forward(self, X:torch.Tensor):

        X = torch.einsum('bij,jk->bik', (X,self.weight))
        adj = self.adj_act(self.Adj)
        X = torch.einsum('ij,bjk->bik', (adj,X))

        if self.bias is not None:
            X = X + self.bias
        
        return X

        

class LearnablePlain_GCNN_Layer(nn.Module):

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
        
        self.gcn = LearnableGraphConvBlock(in_channels, out_channels, time_dim, joints_dim, bias)
        self.act = nn.ReLU()


    def forward(self, X:torch.Tensor):

        X = self.act(self.gcn(X))
        return X

