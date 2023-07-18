import torch
import torch.nn as nn
import numpy as np

from models.gcae.stgcn import ST_GCNN_layer, Graph
from models.gcae.stsgcn import CNN_layer
from models.gcae.gcn import StaticPlain_GCNN_Layer
from models.gcae.learnable_gcn import LearnablePlain_GCNN_Layer


class EncoderSTGCN(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:

        super().__init__()
        # load graph
        graph_args = {'strategy': 'spatial', 'layout': 'openpose', 'headless': False}
        self.graph = Graph(**graph_args)

        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.headless = False

        # build networks
        self.fig_per_seq = 1
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.kernel_size = kernel_size
        self.data_bn = nn.BatchNorm1d(c_in * A.size(1))
        self.dropout = dropout

        self.in_channels = c_in
        self.h_dim = h_dim
        self.channels = channels

        self.build_enc()

        # Edge weighting
        self.ei_enc = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_enc])


    def forward(self, X):
        z = self.encode(X)
        
        return z

    def encode(self, x):
        if self.fig_per_seq == 1:
            if len(x.size()) == 4:
                x = x.unsqueeze(4)
        # Return to (N*M, c, t, v) structure
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_enc, self.ei_enc):
            x, _ = gcn(x, self.A * importance)

        return x


    def build_enc(self):
        """
        Generate and encoder according to a series of dimension factors and strides
        """
        
        in_ch = self.in_channels
        st_gcn_enc = []

        for channel in self.channels:
            st_gcn_enc += [ST_GCNN_layer(in_ch,channel,self.kernel_size,1,self.dropout)]
            in_ch = channel

        st_gcn_enc += [ST_GCNN_layer(in_ch,self.h_dim,self.kernel_size,1,self.dropout)]
        
        self.st_gcn_enc = nn.ModuleList(st_gcn_enc)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



class EncoderLearnablePlainGCN(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.n_joints = n_joints
        
        self.gcns = nn.ModuleList()
        
        in_ch = c_in
        
        for channel in channels:
            self.gcns.append(LearnablePlain_GCNN_Layer(in_ch,channel,n_frames,n_joints,bias=True))
            in_ch = channel
            
        self.gcns.append(LearnablePlain_GCNN_Layer(in_ch,h_dim,n_frames,n_joints,bias=True)) 
        
        
    def forward(self, X:torch.Tensor):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        B,C,T,V = X.size()
        X = X.permute(0,2,3,1).contiguous()
        X = X.view(B,T*V,C).contiguous()

        for gcn in self.gcns:
            X = gcn(X)
        
        X = X.view(B,T,V,X.size(-1)).contiguous()
        X = X.permute(0,3,1,2).contiguous()
        return X



class EncoderStaticPlainGCN(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()

        self.n_frames = n_frames
        self.n_joints = n_joints
        
        links = [(0,1), (0,2), (0,5), (0,6), (1,2), (1,3), (2,4),
                (5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12),
                (11,12), (11,13), (12,14), (13,15), (14,16)]
        Adj = np.zeros((n_joints,n_joints), dtype=np.float32)
        for (i,j) in links:
            Adj[i,j] = 1.0
            Adj[j,i] = 1.0
        Adj = Adj + np.eye(n_joints,n_joints)
        Adj = Adj[np.newaxis,:,np.newaxis,:]
        Adj = np.repeat(np.repeat(Adj,repeats=n_frames,axis=2),repeats=n_frames,axis=0)
        
        for i in range(n_frames-1):
            for j in range(n_joints):
                Adj[i,j,i+1,j] = 1.0
                Adj[i+1,j,i,j] = 1.0

        Adj = Adj.reshape(n_frames*n_joints,n_frames*n_joints)
        Adj = normalize(Adj)
        Adj = torch.tensor(Adj,dtype=torch.float32, requires_grad=False)
        self.register_buffer('Adj', Adj)
        
        self.gcns = nn.ModuleList()
        
        in_ch = c_in
        
        for channel in channels:
            self.gcns.append(StaticPlain_GCNN_Layer(in_ch,channel,n_frames,n_joints,bias=True))
            in_ch = channel
            
        self.gcns.append(StaticPlain_GCNN_Layer(in_ch,h_dim,n_frames,n_joints,bias=True)) 
        
        
    def forward(self, X:torch.Tensor):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        B,C,T,V = X.size()
        X = X.permute(0,2,3,1).contiguous()
        X = X.view(B,T*V,C).contiguous()

        for gcn in self.gcns:
            X = gcn(X, self.Adj)
        
        X = X.view(B,T,V,X.size(-1)).contiguous()
        X = X.permute(0,3,1,2).contiguous()
        return X



class EncoderCNN(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout, channels) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        in_ch = c_in
        
        for channel in channels:
            self.model.append(CNN_layer(in_ch,channel,[1,1],0.0))
            in_ch = channel
            
        self.model.append(CNN_layer(in_ch,h_dim,[1,1],0.0)) 
        
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        return self.model(x)