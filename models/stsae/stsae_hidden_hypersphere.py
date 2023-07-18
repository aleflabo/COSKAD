import torch.nn as nn
import torch

from models.common.components import Encoder, Decoder



class STSAE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSAE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])

        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout, channels)
        self.decoder = Decoder(c_out=c_in, h_dim=h_dim, n_frames=n_frames, n_joints=n_joints, dropout=dropout, channels=channels)
        
        self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)
        self.rev_btlnk = nn.Linear(in_features=latent_dim, out_features=h_dim * n_frames * n_joints)
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.register_buffer('c', torch.zeros(self.latent_dim)) # center c 
        

    def encode(self, x, return_shape=False):
        assert len(x.shape) == 4
        x = x.unsqueeze(4)
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()
            
        x = self.encoder(x)
        N, C, T, V = x.shape
        x = x.view([N, -1]).contiguous()
        x = x.view(N, M, self.h_dim, T, V).permute(0, 2, 3, 4, 1)
        x_shape = x.size()
        x = x.view(N, -1) 
        x = self.btlnk(x)
        
        if return_shape:
            return x, x_shape
        return x
    
    def decode(self, z, input_shape):
        
        z = self.rev_btlnk(z)
        N, C, T, V, M = input_shape
        z = z.view(input_shape).contiguous()
        z = z.permute(0, 4, 1, 2, 3).contiguous()
        z = z.view(N * M, C, T, V)

        z = self.decoder(z)
        
        return z
        
    def forward(self, x):
        hidden_x, x_shape = self.encode(x, return_shape=True) # return the hidden representation of the data x
        x = self.decode(hidden_x, x_shape)
        
        return x, hidden_x