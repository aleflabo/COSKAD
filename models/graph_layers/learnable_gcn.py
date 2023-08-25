import math

import torch
import torch.nn as nn



class LearnableGraphConvBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, n_frames:int, n_joints:int, bias:bool=True) -> None:
        """
        Learnable Graph Convolutional Block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            n_frames (int): number of frames of the sequence
            n_joints (int): number of body joints
            bias (bool, optional): if True, add bias in convolutional operations. Defaults to True.
        """
        
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


    def reset_parameters(self) -> None:
        """
        Initialize the parameters of the model.
        """
        
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.Adj.data.uniform_(0.0,1.0)
        

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, out_channels, time_dim, joints_dim]
        """

        X = torch.einsum('bij,jk->bik', (X,self.weight))
        adj = self.adj_act(self.Adj)
        X = torch.einsum('ij,bjk->bik', (adj,X))

        if self.bias is not None:
            X = X + self.bias
        
        return X

        

class LearnablePlain_GCNN_Layer(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, time_dim:int, joints_dim:int, bias:bool=True) -> None:
        """
        Learnable Graph Convolutional Layer.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            time_dim (int): time dimension of the sequence
            joints_dim (int): number of body joints
            bias (bool, optional): if True, add bias in convolutional operations. Defaults to True.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.joints_dim = joints_dim
        
        self.gcn = LearnableGraphConvBlock(in_channels, out_channels, time_dim, joints_dim, bias)
        self.act = nn.ReLU()


    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, out_channels, time_dim, joints_dim]
        """

        X = self.act(self.gcn(X))
        return X

