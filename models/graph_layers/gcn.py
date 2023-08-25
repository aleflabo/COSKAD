import math

import torch
import torch.nn as nn



class GraphConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, bias:bool=True) -> None:
        """
        Graph Convolutional Block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            bias (bool, optional): if True, add bias. Defaults to True.
        """
        
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


    def forward(self, X:torch.Tensor, Adj:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]
            Adj (torch.Tensor): adjacency matrix of shape

        Returns:
            torch.Tensor: _description_
        """

        X = torch.einsum('bij,jk->bik', (X,self.weight))
        X = torch.einsum('ij,bjk->bik', (Adj,X))

        if self.bias is not None:
            X = X + self.bias
        
        return X

        

class StaticPlain_GCNN_Layer(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, time_dim:int, joints_dim:int, bias:bool=True) -> None:
        """
        Static Graph Convolutional Layer. The adjacency matrix is not learnable.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            time_dim (int): time dimension of the sequence
            joints_dim (int): number of body joints
            bias (bool, optional): if True, add bias. Defaults to True.
        """
        
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.joints_dim = joints_dim
        
        self.gcn = GraphConvBlock(in_channels, out_channels, bias)
        self.act = nn.ReLU()


    def forward(self, X:torch.Tensor, Adj:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]
            Adj (torch.Tensor): adjacency matrix

        Returns:
            torch.Tensor: output tensor of shape [batch_size, out_channels, time_dim, joints_dim]
        """

        X = self.act(self.gcn(X, Adj))
        return X

