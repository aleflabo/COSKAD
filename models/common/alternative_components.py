from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from models.common.components import BaseEncoder
from models.graph_layers.gcn import StaticPlain_GCNN_Layer
from models.graph_layers.learnable_gcn import LearnablePlain_GCNN_Layer
from models.graph_layers.stgcn import Graph, ST_GCNN_layer
from models.graph_layers.stsgcn import CNN_layer



class EncoderSTGCN(BaseEncoder):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Class that implements a Spatial-Temporal Graph Convolutional Encoder (ST-GCN).

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            n_frames (int): number of frames in the sequence, only for compatibility with other encoders
            n_joints (int): number of body joints, only for compatibility with other encoders
            dropout (float): dropout probability
            device (Union[str, torch.DeviceObjType]): device on which to run the model's computations. Defaults to 'cpu'.
        """
        
        self.headless = False
        self.fig_per_seq = 1
        
        # Call the parent class constructor and build the model
        super().__init__(input_dim, layer_channels, hidden_dimension, n_frames, n_joints, dropout, bias=False, device=device)
        
        
    def build_model(self):
        """
        Build the model.
        """
        
        # Set the graph model
        graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': self.headless}
        self.graph = Graph(**graph_args)
        
        # Initialize the adjacency matrix
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False, device=self.device)
        self.register_buffer('A', A)
        
        # Set the kernel size and the batch normalization layer
        spatial_kernel_size = self.A.size(0)
        temporal_kernel_size = 9
        self.kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(self.input_dim * self.A.size(1))
        
        # Build the network
        in_ch = self.input_dim
        st_gcn_enc = []
        stride = 1

        for channel in self.layer_channels:
            st_gcn_enc += [ST_GCNN_layer(in_ch, channel, self.kernel_size, stride, self.dropout)]
            in_ch = channel

        st_gcn_enc += [ST_GCNN_layer(in_ch,self.hidden_dimension, self.kernel_size, stride, self.dropout)]
        
        self.st_gcn_enc = nn.ModuleList(st_gcn_enc)
        
        # Set the importance of each edge
        self.ei_enc = nn.ParameterList([nn.Parameter(torch.ones(self.A.size(), device=self.device)) for _ in self.st_gcn_enc])

    
    def encode(self, X:torch.Tensor) -> torch.Tensor:
        """
        Encode the input tensor.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        if self.fig_per_seq == 1:
            if len(X.size()) == 4:
                X = X.unsqueeze(4)
                
        # Return to (N*M, c, t, v) structure
        N, C, T, V, M = X.size()
        X = X.permute(0, 4, 3, 1, 2).contiguous()
        X = X.view(N * M, V * C, T)
        X = self.data_bn(X)
        X = X.view(N, M, V, C, T)
        X = X.permute(0, 1, 3, 4, 2).contiguous()
        X = X.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_enc, self.ei_enc):
            X, _ = gcn(X, self.A * importance)

        return X

    
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        z = self.encode(X)
        
        return z
    


class EncoderLearnablePlainGCN(BaseEncoder):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True, device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Class that implements a Graph Convolutional Encoder with learnable adjacency matrix.

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional operations. Defaults to True.
            device (Union[str, torch.DeviceObjType]): device on which to run the model's computations. Defaults to 'cpu'.
        """
        
        # Call the parent class constructor and build the model
        super().__init__(input_dim, layer_channels, hidden_dimension, n_frames, n_joints, dropout, bias, device)
        
    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        self.gcns = nn.ModuleList()
        
        in_ch = self.input_dim
        
        for channel in self.layer_channels:
            self.gcns.append(LearnablePlain_GCNN_Layer(in_ch, channel, self.n_frames, self.n_joints, bias=self.bias))
            in_ch = channel
            
        self.gcns.append(LearnablePlain_GCNN_Layer(in_ch, self.hidden_dimension, self.n_frames, 
                                                   self.n_joints, bias=self.bias)) 
        
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        B,C,T,V = X.size()
        X = X.permute(0,2,3,1).contiguous()
        X = X.view(B,T*V,C).contiguous()

        for gcn in self.gcns:
            X = gcn(X)
        
        X = X.view(B,T,V,X.size(-1)).contiguous()
        X = X.permute(0,3,1,2).contiguous()
        return X



class EncoderStaticPlainGCN(BaseEncoder):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True, device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Class that implements a Graph Convolutional Encoder with static adjacency matrix.

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional operations. Defaults to True.
            device (Union[str, torch.DeviceObjType]): device on which to run the model's computations. Defaults to 'cpu'.
        """
        
        # Call the parent class constructor and build the model
        super().__init__(input_dim, layer_channels, hidden_dimension, n_frames, n_joints, dropout, bias, device)
        
        
    def build_model(self) -> None: 
        """
        Build the model.
        """
        
        # Build the adjacency matrix
        Adj = np.zeros((self.n_joints, self.n_joints), dtype=np.float32)
        for (i, j) in self.links:
            Adj[i,j] = 1.0
            Adj[j,i] = 1.0
            
        Adj = Adj + np.eye(self.n_joints, self.n_joints)
        Adj = Adj[np.newaxis, :, np.newaxis, :]
        Adj = np.repeat(np.repeat(Adj, repeats=self.n_frames, axis=2), repeats=self.n_frames, axis=0)
        
        for i in range(self.n_frames-1):
            for j in range(self.n_joints):
                Adj[i,j,i+1,j] = 1.0
                Adj[i+1,j,i,j] = 1.0

        Adj = Adj.reshape(self.n_frames*self.n_joints, self.n_frames*self.n_joints)
        Adj = self.normalize(Adj)
        Adj = torch.tensor(Adj, dtype=torch.float32, requires_grad=False, device=self.device)
        self.register_buffer('Adj', Adj)
        
        self.gcns = nn.ModuleList()
        
        in_ch = self.input_dim
        
        for channel in self.layer_channels:
            self.gcns.append(StaticPlain_GCNN_Layer(in_ch, channel, self.n_frames, self. n_joints, bias=self.bias))
            in_ch = channel
            
        self.gcns.append(StaticPlain_GCNN_Layer(in_ch, self.hidden_dimension, self.n_frames, self. n_joints, bias=self.bias)) 
        
    
    def normalize(self, mx:np.ndarray) -> np.ndarray:
        """
        Normalize the adjacency matrix.

        Args:
            mx (np.ndarray): adjacency matrix

        Returns:
            np.ndarray: normalized adjacency matrix
        """
        
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
        
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        B,C,T,V = X.size()
        X = X.permute(0,2,3,1).contiguous()
        X = X.view(B,T*V,C).contiguous()

        for gcn in self.gcns:
            X = gcn(X, self.Adj)
        
        X = X.view(B,T,V,X.size(-1)).contiguous()
        X = X.permute(0,3,1,2).contiguous()
        return X
    
    
    @property
    def links(self) -> List[tuple]:
        """
        Return the links of the graph.

        Returns:
            List[tuple]: list of links
        """
        
        links = [(0,1), (0,2), (0,5), (0,6), (1,2), (1,3), (2,4),
                    (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12),
                    (11,12), (11,13), (12,14), (13,15), (14,16)]
        return links



class EncoderCNN(BaseEncoder):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True, device:Union[str, torch.DeviceObjType]='cpu') -> None:
        """
        Class that implements a Convolutional Encoder.

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional operations. Defaults to True.
            device (Union[str, torch.DeviceObjType]): device on which to run the model's computations. Defaults to 'cpu'.
        """
        
        # Call the parent class constructor and build the model
        super().__init__(input_dim, layer_channels, hidden_dimension, n_frames, n_joints, dropout, bias, device)
        
    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        self.model = nn.ModuleList()
        
        in_ch = self.input_dim
        kernel_size = (1,1)
        
        for channel in self.layer_channels:
            self.model.append(CNN_layer(in_ch, channel, kernel_size, self.dropout, self.bias))
            in_ch = channel
            
        self.model.append(CNN_layer(in_ch, self.hidden_dimension, kernel_size, self.dropout, self.bias)) 
        
        self.model = nn.Sequential(*self.model)
    
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        return self.model(X)