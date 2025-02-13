import numpy as np
import pypose as pp
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, duration = 1, k_list = [7, 7, 7, 7], c_list = [6, 16, 32, 64, 128], 
                        s_list = [1, 1, 1, 1], p_list = [3, 3, 3, 3]):
        super(CNNEncoder, self).__init__()
        self.duration = duration
        self.k_list, self.c_list, self.s_list, self.p_list = k_list, c_list, s_list, p_list
        layers = []

        for i in range(len(self.c_list) - 1):
            layers.append(torch.nn.Conv1d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(torch.nn.BatchNorm1d(self.c_list[i+1]))
            layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CodeNetMotion(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf    
        self.k_list = [7, 7]
        self.p_list = [3, 3]
        self.s_list = [3, 3]
        self.cnn = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=self.p_list)# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.veldecoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 3))

    def encoder(self, x):
        x = self.cnn(x.transpose(-1, -2)).transpose(-1, -2)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x

    def cov_decoder(self, x):
        cov = torch.exp(self.velcov_decoder(x) - 5.0)
        return cov

    def decoder(self, x):
        vel = self.veldecoder(x)
        return vel
    
    def get_label(self, gt_label):   
        s_idx = (self.k_list[0] - self.p_list[0]) + self.s_list[0] * (self.k_list[1] - 1 -self.p_list[1]) + 1
        select_label = gt_label[:, s_idx::self.s_list[0]* self.s_list[1],:]
        L_out = (gt_label.shape[1] -1 - 1) // self.s_list[0] // self.s_list[1] + 1
        diff = L_out - select_label.shape[1]
        if diff > 0:
            select_label = torch.cat((select_label,gt_label[:,-1:,:].repeat(1,diff , 1)),dim = 1)
        return select_label
    
    def forward(self, data, rot=None):
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature)
        net_vel = self.decoder(feature)
   
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}

    
class CodeNetMotionwithRot(CodeNetMotion):
    def __init__(self, conf):
        super().__init__(conf)
        self.conf = conf    
        self.interval = 9
        self.k_list = [7, 7]
        self.padding_num = 3
        self.s_list = [3, 3]
        
        self.feature_encoder = CNNEncoder(c_list=[6, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.ori_encoder = CNNEncoder(c_list=[3, 32, 64], k_list=self.k_list, s_list=self.s_list, p_list=[3,3])# (N,F/8,64)
        self.gru1 = nn.GRU(input_size = 64, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.gru2 = nn.GRU(input_size = 128, hidden_size =128, num_layers = 1, batch_first = True,bidirectional=True)
        self.fcn1 = nn.Sequential(nn.Linear(128, 128))
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.fcn2 = nn.Sequential(nn.Linear(128, 64))
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        
        self.gelu = nn.GELU()

        
        self.veldecoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))
        self.velcov_decoder = nn.Sequential(nn.Linear(256, 128),nn.GELU(), nn.Linear(128, 3))

    def encoder(self, feature, ori):
        x1 = self.feature_encoder(feature.transpose(-1, -2)).transpose(-1, -2)
        x2 = self.ori_encoder(ori.transpose(-1,-2)).transpose(-1, -2) 
        x = torch.cat([x1, x2], dim = -1)
        
        x = self.fcn2(x)
        x = self.batchnorm2(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.gelu(x)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        return x
    
    def forward(self, data, rot=None):
        assert rot is not None
        feature = torch.cat([data["acc"], data["gyro"]], dim = -1)
        feature = self.encoder(feature, rot)
        net_vel = self.decoder(feature)
   
        #covariance propagation
        cov = None
        if self.conf.propcov:
            cov = self.cov_decoder(feature)
        return {"cov": cov, 'net_vel': net_vel}
