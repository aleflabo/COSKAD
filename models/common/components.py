from typing import List

import torch
import torch.nn as nn
from models.graph_layers.stsgcn import ST_GCNN_layer


class BaseEncoder(nn.Module):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True) -> None:
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
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.input_dim = input_dim
        self.layer_channels = layer_channels
        self.hidden_dimension = hidden_dimension
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.dropout = dropout
        self.bias = bias
        
        # Build the model
        self.build_model()
        
        
    def build_model(self) -> None:
        raise NotImplementedError('This method must be implemented in a child class.')



class Encoder(BaseEncoder):
  
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Encoder (STS-GCN).

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional operations. Defaults to True.
        """
        
        # Call the parent class constructor and build the model
        super().__init__(input_dim, layer_channels, hidden_dimension, 
                         n_frames, n_joints, dropout, bias)
        
    
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        input_channels = self.input_dim
        layer_channels = self.layer_channels + [self.hidden_dimension]
        kernel_size = (1,1)
        stride = 1
        model_layers = nn.ModuleList()
        for channels in layer_channels:
            model_layers.append(
                ST_GCNN_layer(in_channels=input_channels, 
                              out_channels=channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              time_dim=self.n_frames,
                              joints_dim=self.n_joints,
                              dropout=self.dropout,
                              bias=self.bias))
            input_channels = channels
        self.model = nn.Sequential(*model_layers)
    
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, in_channels, time_dim, joints_dim]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, hidden_dimension, time_dim, joints_dim]
        """
        
        return self.model(X)
    
    
    
class Decoder(nn.Module):
    
    def __init__(self, output_dim:int, layer_channels:List[int], hidden_dimension:int, 
                 n_frames:int, n_joints:int, dropout:float, bias:bool=True) -> None:        
        """
        Class that implements a Space-Time-Separable Graph Convolutional Decoder (STS-GCN).

        Args:
            output_dim (int): number of coordinates of the output
            layer_channels (List[int]): list of channel dimension for each layer (in the same order as the encoder's layers)
            hidden_dimension (int): dimension of the hidden layer
            n_frames (int): number of frames of the input pose sequence
            n_joints (int): number of joints of the input pose sequence
            dropout (float): dropout probability
            bias (bool, optional): whether to use bias in the convolutional layers. Defaults to True.
        """
        
        super().__init__()
        
        # Set the model's parameters
        self.output_dim = output_dim
        self.layer_channels = layer_channels[::-1]
        self.hidden_dimension = hidden_dimension
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.dropout = dropout
        self.bias = bias
        
        # Build the model
        self.build_model()
        
        
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        input_channels = self.hidden_dimension
        layer_channels = self.layer_channels + [self.output_dim]
        kernel_size = (1,1)
        stride = 1
        model_layers = nn.ModuleList()
        for channels in layer_channels:
            model_layers.append(
                ST_GCNN_layer(in_channels=input_channels, 
                              out_channels=channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              time_dim=self.n_frames,
                              joints_dim=self.n_joints,
                              dropout=self.dropout,
                              bias=self.bias))
            input_channels = channels
        
        self.model = nn.Sequential(*model_layers)
         

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, hidden_dimension, n_frames, n_joints]

        Returns:
            torch.Tensor: output tensor of shape [batch_size, output_dim, n_frames, n_joints]
        """
        
        return self.model(X)