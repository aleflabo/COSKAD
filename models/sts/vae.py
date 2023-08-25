from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.components import MLP
from power_spherical.distributions import HypersphericalUniform, PowerSpherical

from .ae import STSAE



class STSVAE(STSAE):
    
    def __init__(self, input_dim:int, layer_channels:List[int], hidden_dimension:int, latent_dim:int, 
                 n_frames:int, n_joints:int, encoder_type:str, projector:str, distance:str, dropout:float,
                 bias:bool=True, device:Union[str, torch.DeviceObjType]='cpu', *, projector_hidden_layers:List[int]=None, distribution:str) -> None:
        """
        Class that implements a Space-Time-Separable Graph Convolutional Variational AutoEncoder (STS-GCN): it projects the input in a constrained latent
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
            device (Union[str, torch.DeviceObjType]): device on which to run the model's computations. Defaults to 'cpu'.
            projector_hidden_layers (List[int], optional): list of hidden layers for the projector. Must be specified if the projector is not linear. Defaults to None.
            distribution (str): distribution of the latent space. Must be one of ['normal', 'ps'] (case insensitive)
        """
        
        # Set the model's parameters
        self.distribution = distribution.lower()
        
        # Call the parent class constructor and build the model
        super(STSVAE, self).__init__(input_dim, layer_channels, hidden_dimension, latent_dim,
                                    n_frames, n_joints, encoder_type, projector, distance, dropout, bias, device, 
                                    projector_hidden_layers=projector_hidden_layers) 
        
        
    def build_model(self) -> None:
        """
        Build the model.
        """
        
        super().build_model()
        
        # expected value of the distribution of the normal data
        if self.distribution == 'normal':
            self.register_buffer('mean_vector', torch.zeros((1, self.latent_dim), device=self.device))
        # threshold of the cosine distance 
        self.register_buffer('threshold_dist', torch.tensor(0, dtype=torch.float32, device=self.device)) 
        

    def encode(self, X:torch.Tensor, return_shape:bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Encode the input in the latent space.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_dim, n_frames, n_joints]
            return_shape (bool, optional): if True, return the shape of the input tensor before the bottlenek. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]: latent representation and (optionally) shape of the input tensor before the bottleneck
        """
        
        Z, X_shape = super().encode(X, return_shape=True)
        
        # borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        # Compute mean
        Z_mean = self.fc_mean(Z)
        if self.distribution == 'ps':
            Z_mean = Z_mean / torch.norm(Z_mean, dim=-1, keepdim=True)
        
        # Compute variance or concentration
        # the `+ 1` prevent collapsing behaviors
        Z_var = F.softplus(self.fc_var(Z)) + 1
        
        if return_shape:
            return Z_mean, Z_var, X_shape
        
        return Z_mean, Z_var
    
    
    def reparameterize(self, Z_mean:torch.Tensor, Z_var:torch.Tensor) -> Tuple[torch.distributions.Distribution]:
        """
        Reparameterization trick to sample from the latent space.
        Borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py

        Args:
            Z_mean (torch.Tensor): mean of the latent space
            Z_var (torch.Tensor): variance of the latent space
            
        Returns:
            Tuple[torch.distributions.Distribution]: distributions of the latent space
        """

        if self.distribution == 'normal':
            q_Z = torch.distributions.normal.Normal(Z_mean, Z_var)
            p_Z = torch.distributions.normal.Normal(torch.zeros_like(Z_mean, device=self.device), torch.ones_like(Z_var, device=self.device))
        elif self.distribution == 'ps':
            q_Z = PowerSpherical(loc=Z_mean, scale=torch.squeeze(Z_var, dim=-1))
            p_Z = HypersphericalUniform(self.latent_dim - 1, device=self.device)
        
        return q_Z, p_Z
    
    
    def forward(self, X:torch.Tensor) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]: 
        """
        Forward pass of the model.

        Args:
            X (torch.Tensor): input tensor of shape [batch_size, input_dim, n_frames, n_joints]

        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]: latent representation, reconstructed input and distributions of the latent space 
        """
        
        Z_mean, Z_var, input_shape = self.encode(X, return_shape=True)
        q_Z, p_Z = self.reparameterize(Z_mean, Z_var)
        Z = q_Z.rsample()
        X = self.decode(Z, input_shape=input_shape)
        
        return Z, X, (q_Z, p_Z, Z_var)
    
    
    def _set_projector_type(self) -> None:
        """
        Set the projector type.

        Raises:
            ValueError: if the distribution is not supported or if the projector type is not supported.
        """
        
        input_size = self.hidden_dimension * self.n_frames * self.n_joints
        
        if self.projector == 'mlp':
            assert self.projector_hidden_layers is not None, 'projector_hidden_layers must be specified if the projector is not linear'
            self.btlnk = MLP(input_size=input_size, output_size=self.latent_dim, hidden_size=[self.latent_dim])
            input_size = self.latent_dim
            
        else:
            self.btlnk = nn.Identity()
            assert self.projector == 'linear', f'Projector type {self.projector} not supported.'
        
        # Mean of the normal distribution or mean of the PowerSpherical distribution
        self.fc_mean = nn.Linear(in_features=input_size, out_features=self.latent_dim)
        
        # code borrowed from https://github.com/nicola-decao/s-vae-pytorch/blob/master/examples/mnist.py
        if self.distribution == 'normal':
            var_out_features = self.latent_dim
            # compute mean and std of the normal distribution
            
        elif self.distribution == 'ps':
            var_out_features = 1
                
        else:
            raise ValueError(f'Distribution {self.distribution} not supported.')

        # Variance of the normal distribution or concentration of the PowerSpherical distribution
        self.fc_var =  nn.Linear(in_features=input_size, out_features=var_out_features)
        
