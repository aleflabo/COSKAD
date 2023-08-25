"""
Graph definitions, based on awesome previous work by https://github.com/yysijie/st-gcn
"""

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn



class Graph:

    def __init__(self, layout:str='openpose', strategy:str='spatial', headless:bool=False, max_hop:int=1) -> None:
        """ 
        The Graph to models the skeletons extracted by the openpose

        Args:
            strategy (string): must be one of the follow candidates
            - uniform: Uniform Labeling
            - distance: Distance Partitioning
            - spatial: Spatial Configuration
            For more information, please refer to the section 'Partition Strategies'
                in our paper (https://arxiv.org/abs/1801.07455).

            layout (string): must be one of the follow candidates
            - openpose: Is consists of 17 joints. For more information, please
                refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
            - ntu-rgb+d: Is consists of 25 joints. For more information, please
                refer to https://github.com/shahroudy/NTURGB-D

            max_hop (int): the maximal distance between two connected nodes
        """
        
        self.headless = headless
        self.max_hop = max_hop
        
        self.get_edge(layout)
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)


    def get_edge(self, layout:str) -> None:
        """
        Gets the edge of the graph.

        Args:
            layout (str): layout of the graph. Must be one of the follow candidates
            - openpose: Is consists of 17 joints.
            - ntu-rgb+d: Is consists of 25 joints.

        Raises:
            ValueError: if the layout is not supported
        """
        
        if layout == 'openpose':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0,1), (0,2), (0,5), (0,6), (1,2), (1,3), (2,4),
                            (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12),
                            (11,12), (11,13), (12,14), (13,15), (14,16)]
            self.edge = self_link + neighbor_link
            self.center = 1
            
        elif layout == 'ntu-rgb+d':
            
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1

        else:
            raise ValueError("Do Not Exist This Layout.")


    def get_adjacency(self, strategy:str) -> None:
        """
        Gets the adjacency matrix of the graph.

        Args:
            strategy (str): must be one of the follow candidates
            - uniform: Uniform Labeling
            - distance: Distance Partitioning
            - spatial: Spatial Configuration

        Raises:
            ValueError: if the strategy is not supported
        """
        
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
            
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
            
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
            
        else:
            raise ValueError("Do Not Exist This Strategy")


    def get_hop_distance(self, num_node:int, edge:Iterable, max_hop:int=1) -> np.ndarray:
        """
        Computes the hop distance between any two nodes in a graph.

        Args:
            num_node (int): number of nodes in the graph
            edge (Iterable): iterable of edges in the graph
            max_hop (int, optional): maximum number of hops. Defaults to 1.

        Returns:
            np.ndarray: the hop distance matrix
        """
        
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis


    def normalize_digraph(A:np.ndarray) -> np.ndarray:
        """
        Normalizes the adjacency matrix of a directed graph.

        Args:
            A (np.ndarray): adjacency matrix of a directed graph

        Returns:
            np.ndarray: normalized adjacency matrix
        """
        
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD



class ConvTemporalGraphical(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, t_kernel_size:int=1, t_stride:int=1, bias:bool=True) -> None:
        """
        The basic module for applying a graph convolution.
        
        Args:
            in_channels (int): number of channels in the input sequence data
            out_channels (int): number of channels produced by the convolution
            kernel_size (int): size of the graph convolving kernel
            t_kernel_size (int): size of the temporal convolving kernel
            t_stride (int, optional): stride of the temporal convolution. Defaults to 1
            bias (bool, optional): if True, adds a learnable bias to the output. Defaults to True
        """
        
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              stride=(t_stride, 1),
                              dilation=(1, 1),
                              bias=bias)
        

    def forward(self, x:torch.Tensor, adj:torch.Tensor) -> Tuple[torch.Tensor]:
        assert adj.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj))

        return x.contiguous(), adj



class STConvBlock(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=1, dropout:float=0.0) -> None:
        """
        Spatial-temporal convolutional block.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (int, optional): kernel size of the convolutional layer. Defaults to 1.
            dropout (float, optional): dropout probability. Defaults to 0.0.
        """
        
        super().__init__()
        
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multi_adj = kernel_size > 1

        self.g_conv = ConvTemporalGraphical(in_channels, out_channels, kernel_size)


    def forward(self, inp:torch.Tensor, adj:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of the layer.

        Args:
            inp (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]
            adj (torch.Tensor): adjacency matrix 

        Returns:
            Tuple[torch.Tensor]: output tensor of shape [batch_size, in_channels, time_dim, joints_dim] and adjacency matrix
        """
        
        return self.g_conv(inp, adj)



class ST_GCNN_layer(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int=1, dropout:float=0., 
                 act:nn.Module=None, out_bn:bool=True, out_act:bool=True, residual:bool=True, headless:bool=False) -> None:
        """
        Spatial-temporal GCN.

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            kernel_size (int): kernel size of the convolutional layer
            stride (int, optional): stride of the convolution. Defaults to 1.
            dropout (float, optional): dropout probability. Defaults to 0..
            act (nn.Module, optional): activation function; if None, nn.ReLU is used. Defaults to None.
            out_bn (bool, optional): if True, apply BatchNorm2D to the output. Defaults to True.
            out_act (bool, optional): if True, apply the activation function to the output. Defaults to True.
            residual (bool, optional): if True, add residuals. Defaults to True.
            headless (bool, optional): if True, exclude some joints. Defaults to False.
        """
        
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.headless = headless
        self.out_act = out_act
        self.act = nn.ReLU(inplace=True) if act is None else act
        self.gcn = STConvBlock(in_channels, out_channels, kernel_size=kernel_size[1], dropout=dropout,
                               headless=self.headless)

        if out_bn:
            bn_layer = nn.BatchNorm2d(out_channels)
        else:
            bn_layer = nn.Identity() 

        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 self.act,
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 bn_layer,
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda _: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
            

    def forward(self, x:torch.Tensor, adj:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]
            adj (torch.Tensor): adjacency matrix

        Returns:
            Tuple[torch.Tensor]: output tensor of shape [batch_size, in_channels, time_dim, joints_dim] and adjacency matrix
        """
        
        res = self.residual(x)
        x, adj = self.gcn(x, adj)
        x = self.tcn(x) + res
        if self.out_act:
            x = self.act(x)

        return x, adj