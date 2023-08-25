from typing import List, Tuple, Union

import torch
import torch.nn as nn
from models.common.alternative_components import (EncoderLearnablePlainGCN,
                                                  EncoderStaticPlainGCN,
                                                  EncoderSTGCN)
from models.common.components import MLP, Decoder, Encoder



class STSE(nn.Module):
    
    encoder_classes = {'sts_gcn': Encoder, 'st_gcn': EncoderSTGCN, 'learnable_gcn': EncoderLearnablePlainGCN, 'static_gcn': EncoderStaticPlainGCN}
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, latent_dim:int, 
                 n_frames:int, n_joints:int, encoder_type:str, projector:str, distance:str, dropout:float, bias:bool=True,
                 device:Union[str, torch.DeviceObjType]='cpu', *, projector_hidden_layers:List[int]=None) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Encoder (STS-GCN) that projects the input in a constrained latent
        space, forcing the latent representations to be close to a data-driven center `c` according to a certain distance metric.

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            latent_dim (int): dimension of the latent space
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            encoder_type (str): type of the encoder. Must be one of ['sts_gcn', 'st_gcn', 'learnable_gcn', 'static_gcn'] (case insensitive)
            projector (str): type of the projector. Must be one of ['linear', 'mlp'] (case insensitive)
            distance (str): distance metric to use. Must be one of ['euclidean', 'mahalanobis'] (case insensitive)
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional and linear operations. Defaults to True.
            device (Union[str, torch.DeviceObjType]): device on which to run the model. Defaults to 'cpu'.
            projector_hidden_layers (List[int], optional): list of hidden layers for the projector. Must be specified if the projector is not linear. Defaults to None.
        """
        
        super(STSE, self).__init__()
        
        # Set the model's parameters
        self.input_dim = input_dim
        self.layer_channels = layer_channels
        self.hidden_dimension = hidden_dimension
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.n_joints = n_joints
        self.encoder_type = encoder_type.lower()
        self.projector = projector.lower()
        self.projector_hidden_layers = projector_hidden_layers
        self.distance = distance.lower()
        self.dropout = dropout
        self.bias = bias
        self.device = device
        
        # Build the model
        self.build_model()
        
    
    def build_model(self) -> None:
        
        # Set the encoder
        self._set_encoder_type()
        
        # Set the projector
        self._set_projector_type()
        
        # Set the center c which will be initialized before the training starts
        self.register_buffer('c', torch.zeros(self.latent_dim))
        
        # Set the inverse covariance matrix if distance is 'mahalanobis'
        if self.distance == 'mahalanobis':
            self.register_buffer('inv_cov_matrix', torch.zeros((self.latent_dim, self.latent_dim)))
            

    def encode(self, X:torch.Tensor, return_shape:bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Encode the input data X. Optionally return the shape of the input (useful for the decoder)

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_dim, n_frames, n_joints]
            return_shape (bool, optional): whether to return the shape of the input. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: the hidden representation of the input data X. If return_shape is True, also return the shape of the input
        """
        
        assert len(X.shape) == 4, f'Input tensor must have shape [batch_size, input_dim, n_frames, n_joints]. Got {X.shape}'
        X = X.unsqueeze(4)
        B, C, T, V, M = X.size()

        X = X.permute(0, 4, 3, 1, 2).contiguous()
        X = X.view(B * M, V, C, T).permute(0,2,3,1).contiguous()
            
        X = self.encoder(X)
        B, C, T, V = X.shape
        X = X.view([B, -1]).contiguous()
        X = X.view(B, M, self.hidden_dimension, T, V).permute(0, 2, 3, 4, 1)
        X_shape = X.size()
        X = X.view(B, -1) 
        X = self.btlnk(X)
        
        if return_shape:
            return X, X_shape
        return X
    
        
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_dim, n_frames, n_joints]

        Returns:
            torch.Tensor: the representation of the input data X in the latent space. Shape [batch_size, latent_dim]
        """
        
        X = self.encode(X)
        
        return X
    
    
    def _set_encoder_type(self) -> None:
        """
        Set the encoder type.

        Raises:
            ValueError: if the encoder type is not supported
        """

        if self.encoder_type in self.encoder_classes.keys():
            self.encoder = self.encoder_classes[self.encoder_type](input_dim=self.input_dim, 
                                                                   layer_channels=self.layer_channels, 
                                                                   hidden_dimension=self.hidden_dimension, 
                                                                   n_frames=self.n_frames, 
                                                                   n_joints=self.n_joints, 
                                                                   dropout=self.dropout, 
                                                                   bias=self.bias,
                                                                   device=self.device)
        else:
            raise ValueError(f'Encoder type {self.encoder_type} not supported.')    

        print(f'Encoder type: {self.encoder_type}')
        
        
    def _set_projector_type(self) -> None:
        """
        Set the projector type.

        Raises:
            ValueError: if the projector type is not supported
        """
        
        input_size = self.hidden_dimension * self.n_frames * self.n_joints
        if self.projector == 'linear':
            self.btlnk = nn.Linear(in_features=input_size, out_features=self.latent_dim, bias=self.bias)
            
        elif self.projector == 'mlp':
            assert self.projector_hidden_layers is not None, 'projector_hidden_layers must be specified if the projector is not linear'
            self.btlnk = MLP(input_size=input_size, output_size=self.latent_dim, hidden_size=self.projector_hidden_layers, bias=self.bias, device=self.device)
            
        else:
            raise ValueError(f'Projector type {self.projector} not supported.')
        
        
        
class STSAE(STSE):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, latent_dim:int, 
                 n_frames:int, n_joints:int, encoder_type:str, projector:str, distance:str, dropout:float, bias:bool=True, 
                 device:Union[str, torch.DeviceObjType]='cpu', *, projector_hidden_layers:List[int]=None) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional AutoEncoder (STS-GCN): it projects the input in a constrained latent
        space, forcing the latent representations to be close to a data-driven center `c` according to a certain distance metric, and then reconstructs
        the input from the latent representation.

        Args:
            input_dim (int): number of input channels
            layer_channels (List[int]): list of channels for each layer
            hidden_dimension (int): hidden dimension of the encoder
            latent_dim (int): dimension of the latent space
            n_frames (int): number of frames in the sequence
            n_joints (int): number of body joints
            encoder_type (str): type of the encoder. Must be one of ['sts_gcn', 'st_gcn', 'learnable_gcn', 'static_gcn'] (case insensitive)
            projector (str): type of the projector. Must be one of ['linear', 'mlp'] (case insensitive)
            distance (str): distance metric to use. Must be one of ['euclidean', 'mahalanobis'] (case insensitive)
            dropout (float): dropout probability
            bias (bool, optional): if True, add bias in the convolutional and linear operations. Defaults to True.
            device (Union[str, torch.DeviceObjType]): device on which to run the model. Defaults to 'cpu'.
            projector_hidden_layers (List[int], optional): list of hidden layers for the projector. Must be specified if the projector is not linear. Defaults to None.
        """
        
        # Call the parent class constructor and build the model
        super(STSAE, self).__init__(input_dim, layer_channels, hidden_dimension, latent_dim,
                                    n_frames, n_joints, encoder_type, projector, distance, dropout, device, bias,
                                    projector_hidden_layers=projector_hidden_layers) 
        
        
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        super().build_model()
        self.rev_btlnk = nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dimension * self.n_frames * self.n_joints)
        self._set_decoder_type()        
        
    
    def decode(self, Z:torch.Tensor, input_shape:Tuple[int]) -> torch.Tensor:
        """
        Decode the latent representation Z.

        Args:
            Z (torch.Tensor): latent representation of shape [batch_size, latent_dim]
            input_shape (Tuple[int]): shape of the input data. Must be [batch_size, 1, hidden_dimension, n_frames, n_joints]

        Returns:
            torch.Tensor: the reconstructed input data of shape [batch_size, input_dim, n_frames, n_joints]
        """
        
        Z = self.rev_btlnk(Z)
        B, C, T, V, M = input_shape
        Z = Z.view(input_shape).contiguous()
        Z = Z.permute(0, 4, 1, 2, 3).contiguous()
        Z = Z.view(B * M, C, T, V)

        Z = self.decoder(Z)
        
        return Z
    
        
    def forward(self, X:torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_dim, n_frames, n_joints]

        Returns:
            Tuple[torch.Tensor]: the representation of the input data X in the latent space and the reconstructed input data X.
        """
        
        # Encode the input data
        Z, X_shape = self.encode(X, return_shape=True)
        
        # Decode the latent representation
        X = self.decode(Z, X_shape)
        
        return Z, X

        
    def _set_decoder_type(self) -> None:
        """
        Set the decoder type.

        Raises:
            ValueError: if the encoder type is not supported
        """
        
        if self.encoder_type == 'sts_gcn':
            self.decoder = Decoder(self.input_dim, self.layer_channels, self.hidden_dimension, self.n_frames, self.n_joints, self.dropout, self.bias)
            
        else:
            raise ValueError(f'No decoder available for encoder type {self.encoder_type}.')
        