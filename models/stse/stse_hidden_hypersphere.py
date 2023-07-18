import torch.nn as nn
import torch

from models.common.components import Encoder
from models.common.alternative_components import EncoderStaticPlainGCN, EncoderSTGCN, EncoderLearnablePlainGCN



class STSE(nn.Module):
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, **kwargs) -> None:
        super(STSE, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)
        channels = kwargs.get('channels', [128,64,128])
        distance = kwargs.get('distance', 'euclidean')
        projector = kwargs.get('projector', 'linear')
        encoder_type = kwargs.get('encoder_type', 'STS_GCN')

        if encoder_type == 'STS_GCN':
            print('Using STS-GCN')
            self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout, channels)
        elif encoder_type == 'ST_GCN':
            print('Using ST-GCN')
            self.encoder = EncoderSTGCN(c_in, h_dim, n_frames, n_joints, dropout, channels)
        elif encoder_type == 'Learnable_GCN':
            print('Using Learnable-GCN')
            self.encoder = EncoderLearnablePlainGCN(c_in,h_dim,n_frames,n_joints,dropout,channels)
        elif encoder_type == 'Static_GCN':
            print('Using Static-GCN')
            self.encoder = EncoderStaticPlainGCN(c_in, h_dim, n_frames, n_joints, dropout, channels)
        
        if projector == 'linear':
            self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)
        else:
            self.btlnk = MLP(input_size = h_dim * n_frames * n_joints, hidden_size=[16,16])
        
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.distance = distance
        self.register_buffer('c', torch.zeros(self.latent_dim)) # center c which will be initialized before the training start
        if self.distance == 'mahalanobis':
            self.register_buffer('inv_cov_matrix', torch.zeros((self.latent_dim, self.latent_dim)))
        

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
    
        
    def forward(self, x):
        x, x_shape = self.encode(x, return_shape=True) # return the hidden representation of the data x
        
        return x
    
class MLP(nn.Module):
    """MLP class for projector and predictor."""

    def __init__(self, input_size, hidden_size, hyper=False, bias=True):
        super().__init__()

        n_layers = len(hidden_size)
        layer_list = []

        for idx, next_dim in enumerate(hidden_size):

            if idx == n_layers-1:
                layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                # if bias is False:
                #     layer_list.append(nn.BatchNorm1d(next_dim, affine=False))
            else:
                # keep dim as in literature
                layer_list.append(nn.Linear(input_size, next_dim, bias=bias))
                layer_list.append(nn.BatchNorm1d(next_dim))
                layer_list.append(nn.ReLU(inplace=True))
                input_size = next_dim

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
