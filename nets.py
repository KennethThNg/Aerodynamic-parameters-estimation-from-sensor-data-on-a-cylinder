import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class ResidualTCN(nn.Module):
    def __init__(self, time_len, out_dim, in_chan, out_chan,
                  kernel_pool=2, stride_pool=2, kernel_conv=5, stride_conv=1):
        super(ResidualTCN, self).__init__()
        #dimension init:
        self.time_len = time_len #T
        self.out_dim = out_dim #H
        self.in_chan = in_chan #C
        self.out_chan = out_chan #C'

        self.kernel_pool = kernel_pool
        self.stride_pool = stride_pool

        self.kernel_conv = kernel_conv
        self.stride_conv = stride_conv

        self.out_pool_dim = int((self.time_len - self.kernel_pool)/self.kernel_pool + 1) #Tp
        self.out_conv_dim = int((self.out_pool_dim - self.kernel_conv)/self.stride_conv + 1) #T'

        self.in_fc_dim = self.out_conv_dim*self.out_chan #T'C'=C'T'

        #layers
        self.max_pool = nn.MaxPool2d(kernel_size=[self.kernel_pool, 1], stride=[self.stride_pool, 1])
        self.conv = nn.Conv2d(in_channels=self.in_chan, out_channels=self.out_chan, kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv,1])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=self.in_fc_dim, out_features=self.out_dim)

        #init weight
        self.reset_weight()

    def forward(self, x): #[B,C,T,D]
        batch_size = x.shape[0]
        x_dim = x.shape[3]
        out = self.max_pool(x) #[B, C, Tp, D]
        out = self.conv(out) #[B, C', T', D]
        out = self.relu(out)
        out = out.permute(0,3,1,2) #[B, D, C', T']Oui faisons
        out = out.view(batch_size, x_dim, self.in_fc_dim) #[B, D, C'T']
        out = self.fc(out) #[B, C, H]
        return out

    def reset_weight(self):
        self.conv.weight.data.normal_(std=0.1)
        self.fc.weight.data.normal_(std=0.1)

class TemporalConvNet(nn.Module):
    def __init__(self, in_dim, time_len, hid_dim, out_dim, in_chan=1, model_hyperparams=None):
        super(TemporalConvNet, self).__init__()
        default_hyperparams = {
            'hid_chan1': 6, #C1
            'hid_chan2': 8, #C2
            'hid_chan3': 10, #C3

            'kernel_conv': 5,
            'stride_conv': 1,

            'kernel_pool': 2,
            'stride_pool': 2
        }

        # Store the hyperparameters
        self._update_params(default_hyperparams, model_hyperparams)

        #dimension layers
        self.in_dim = in_dim #D
        self.time_len = time_len #T
        self.hid_dim = hid_dim #H
        self.out_dim = out_dim #Y

        self.in_chan = in_chan #C

        self.in_conv1 = self.time_len - self.kernel_conv + 1 #T1
        self.in_conv2 = self.in_conv1 - self.kernel_conv + 1 #T2
        self.in_conv3 = self.in_conv2 - self.kernel_conv + 1 #T2

        self.in_fc_dim = self.in_conv3*self.hid_chan3 #T3*C3

        #Layers
        self.conv1 = nn.Conv1d(in_channels=self.in_chan, out_channels=self.hid_chan1,
                               kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv,1])

        self.conv2 = nn.Conv1d(in_channels=self.hid_chan1, out_channels=self.hid_chan2,
                               kernel_size=[self.kernel_conv, 1], stride=[self.stride_conv,1])

        self.conv3 = nn.Conv1d(in_channels=self.hid_chan2, out_channels=self.hid_chan3,
                               kernel_size=[self.kernel_conv,1], stride=[self.stride_conv,1])

        self.resnet1 = ResidualTCN(time_len=self.in_conv1, out_dim=self.hid_dim, in_chan=self.hid_chan1, out_chan=self.hid_chan2,
                                   kernel_pool=self.kernel_pool, stride_pool=self.stride_pool, kernel_conv=self.kernel_conv, stride_conv=self.stride_conv)

        self.resnet2 = ResidualTCN(time_len=self.in_conv2, out_dim=self.hid_dim, in_chan=self.hid_chan2, out_chan=self.hid_chan3,
                                   kernel_pool=self.kernel_pool, stride_pool=self.stride_pool, kernel_conv=self.kernel_conv, stride_conv=self.stride_conv)


        self.relu = nn.ReLU()

        self.fc = nn.Linear(in_features=self.in_fc_dim, out_features=self.hid_dim)

        self.fcb = nn.Linear(in_features=3*self.hid_dim, out_features=self.time_len)

        self.fcd = nn.Linear(in_features=self.in_dim, out_features=self.out_dim)

        #init weight
        self.reset_weights()

    def forward(self, x): #[B, C, T, D]
        batch_size = x.shape[0]
        x_dim = x.shape[3]
        out1_1 = self.conv1(x) #[B, C1, T1, D]
        out1_1 = self.relu(out1_1)

        out1_2 = self.resnet1(out1_1) #[B, D, H]

        out2_1 = self.conv2(out1_1) #[B, C2, T2, D]

        out2_2 = self.resnet2(out2_1) #[B, D, H]

        out3_1 = self.conv3(out2_1) #[B, C3, T3, D]
        out3_1 = self.relu(out3_1)
        out3_1 = out3_1.permute(0,3,1,2) #[B, D, C3, T3]
        out3_1 = out3_1.view(batch_size, x_dim, self.in_fc_dim) #[B, D, C3*T3]
        out3_2 = self.fc(out3_1) #[B, D, H]

        out_cat = torch.cat((out1_2, out2_2, out3_2), dim=2) #[B, D, 3H]
        out4 = self.fcb(out_cat) #[B, D, T]
        out4 = out4.permute(0,2,1) #[B, T, D]
        out = self.fcd(out4) #[B, T, Y]
        return out

    def reset_weights(self):
        self.resnet1.reset_weight()
        self.resnet2.reset_weight()
        self.conv1.weight.data.normal_(std=0.1)
        self.conv2.weight.data.normal_(std=0.1)
        self.conv3.weight.data.normal_(std=0.1)
        self.fc.weight.data.normal_(std=0.1)
        self.fcb.weight.data.normal_(std=0.1)
        self.fcd.weight.data.normal_(std=0.1)


    def _set_params(self, params):
        for k in params.keys():
            self.__setattr__(k, params[k])

    def _update_params(self, prev_params, new_params):
        if new_params:
            params = update_param_dict(prev_params, new_params)
        else:
            params = prev_params
        self._set_params(params)


class Wavenet(nn.Module):
    def __init__(self, conv_size, nb_of_measurements, nb_of_outputs):
        super(Wavenet, self).__init__()        
        self.in_channels = nb_of_measurements
        self.out_channels = nb_of_outputs
        self.nb_timesteps = conv_size
        self.nb_layers = int(np.log2(self.nb_timesteps))
        self.conv = nn.ModuleList()
        self.layer_change = int(np.floor((self.in_channels - self.out_channels) / int(np.log2(self.nb_timesteps))))
        in_c = self.in_channels
        out_c = self.in_channels - self.layer_change
        for i in range(self.nb_layers):
            self.conv.append(nn.Conv1d(in_channels=in_c, 
                                  out_channels=out_c, 
                                  kernel_size=2, 
                                  stride=2))

            if (((out_c >= self.out_channels - self.layer_change)
                 & (self.in_channels >= self.out_channels))
                    |  ((out_c <= self.out_channels + self.layer_change)
                        & (self.in_channels < self.out_channels))):
                if i == self.nb_layers-2:
                    in_c = out_c
                    out_c = self.out_channels
                else:
                    in_c = in_c - self.layer_change 
                    out_c = out_c - self.layer_change
            else:
                in_c = self.out_channels
                out_c = self.out_channels
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(self.nb_layers):
            x = self.conv[i](x)
        x = x.squeeze()
        return x
    
    def reset_parameters(self):
        for i in range(self.nb_layers):
            self.conv[i].reset_parameters()