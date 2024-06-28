import torch
from torch import nn
import normflows as nf

class ContextCNN(nn.Module):
    def __init__(self, num_channels, context_size, dropout_rate=0.2):
        # call constructor from superclass
        super().__init__()
        # define network layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=(2,2), stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,2), stride=1, padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flat = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(32,16),
            nn.PReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout_rate),
            nn.Linear(16, context_size),
        )


    def forward(self, x):
        # define forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class NFM(nn.Module):
    def __init__(self, context_size, K = 3, hidden_units = 128, num_blocks = 2):
        super().__init__()
        latent_size = 2
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=context_size, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
#        self.num_samples = num_samples

    def forward(self, samples, context):
        num_samples = samples.shape[1]
        context_rep = torch.repeat_interleave(context, num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l

class Discriminator(nn.Module):
    def __init__(self, input_size=8, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y


class Combined(nn.Module):
    def __init__(self,num_channels, context_size, num_samples, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextCNN(num_channels,context_size,dropout_rate)
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=context_size, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        self.num_samples = num_samples
        
    def forward(self, samples, x):
        context = self.context_encoder(x)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, self.num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class Combined_2branches(nn.Module):
    def __init__(self,num_channels_diff, num_channels_mt, context_size, num_samples, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()

        self.context_encoder_diff = ContextCNN(num_channels_diff,context_size,dropout_rate)
        self.context_encoder_mt = ContextCNN(num_channels_mt,context_size,dropout_rate)
        
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=context_size*2, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        self.num_samples = num_samples
        
    def forward(self, samples, x_diff, x_mt):
        context_diff = self.context_encoder_diff(x_diff)
        context_mt = self.context_encoder_mt(x_mt)
        context = torch.cat((context_diff,context_mt),1)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, self.num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class ContextMLP(nn.Module):
    def __init__(self, num_channels, context_size, dropout_rate=0.2):
        # call constructor from superclass
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels,64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64,32),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, context_size),
        )

    def forward(self, x):
        # define forward pass
        x = self.fc(x)
        return x


class CombinedMLP(nn.Module):
    def __init__(self,num_channels, context_size, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextMLP(num_channels,context_size,dropout_rate)
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=context_size, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        
    def forward(self, samples, num_samples, x):
        context = self.context_encoder(x)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class ContextMLP_withMaps(nn.Module):
    def __init__(self, num_channels, context_size, dropout_rate=0.2):
        # call constructor from superclass
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels,64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64,32),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, context_size),
        )

    def forward(self, x, num_mri_raw):
        # define forward pass
        raw_mri=x[:,:num_mri_raw]
        maps=x[:,num_mri_raw:]
        x = self.fc(raw_mri)
        return torch.cat([x,maps],dim=1)


class CombinedMLP_withMaps(nn.Module):
    def __init__(self,num_channels, num_mri_raw, context_size, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextMLP_withMaps(num_mri_raw,context_size,dropout_rate)
        self.num_mri_raw = num_mri_raw
        self.num_param_maps = num_channels-num_mri_raw
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        self.total_context = context_size+self.num_param_maps
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=self.total_context, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        
    def forward(self, samples, num_samples, x):
        context = self.context_encoder(x, self.num_mri_raw)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class ContextMLP_withMaps_larger(nn.Module):
    def __init__(self, num_channels, num_raw, context_size, dropout_rate=0.2):
        # call constructor from superclass
        super().__init__()
        self.num_raw = num_raw
        self.num_maps = num_channels-num_raw
        self.fc = nn.Sequential(
            nn.Linear(num_raw,64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64,32),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, context_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.num_maps,16),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16,16),
            nn.PReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 8),
        )

    def forward(self, x):
        # define forward pass
        raw_mri=x[:,:self.num_raw]
        maps=x[:,self.num_raw:]
        x1 = self.fc(raw_mri)
        x2 = self.fc2(maps)
        return torch.cat([x1,x2],dim=1)

class CombinedMLP_withMaps_larger(nn.Module):
    def __init__(self,num_channels, num_mri_raw, context_size, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextMLP_withMaps_larger(num_channels,num_mri_raw,context_size,dropout_rate)
        self.num_mri_raw = num_mri_raw
        self.num_param_maps = num_channels-num_mri_raw
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        self.total_context = context_size+8
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=self.total_context, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        
    def forward(self, samples, num_samples, x):
        context = self.context_encoder(x)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class ContextMLP_withMaps_adjacent(nn.Module):
    def __init__(self, num_channels, num_mri_raw, context_size, dropout_rate=0.2):
        # call constructor from superclass
        super().__init__()
        self.num_raw = num_mri_raw
        self.num_maps = num_channels-num_mri_raw
        self.fc = nn.Sequential(
            nn.Linear(self.num_raw,64),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64,32),
            nn.PReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, context_size),
        )

    def forward(self, x):
        # define forward pass
        context_list = []
        for ii in range(x.shape[2]): #iterate over voxels
            raw_mri=x[:,:self.num_raw,ii]
            maps=x[:,self.num_raw:,ii]
            raw_context = self.fc(raw_mri)
            context_list.append(torch.cat([raw_context,maps],dim=1))
        full_context = torch.cat(context_list,dim=1)
        return full_context

class CombinedMLP_withMaps_adjacent(nn.Module):
    def __init__(self,num_channels, num_mri_raw, num_voxels, context_size, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextMLP_withMaps_adjacent(num_channels,num_mri_raw,context_size,dropout_rate)
        self.num_mri_raw = num_mri_raw
        self.num_param_maps = num_channels-num_mri_raw
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        self.total_context = (context_size+self.num_param_maps)*num_voxels
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=self.total_context, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm
        
    def forward(self, samples, num_samples, x):
        context = self.context_encoder(x)
        # trick to use all samples at once
        context_rep = torch.repeat_interleave(context, num_samples, 0)
        l = self.flow_model.forward_kld(samples, context_rep)
        return l


class CombinedMLP_withMaps_auxTasks(nn.Module):
    def __init__(self,num_channels, num_mri_raw, context_size, K = 3, hidden_units = 128, num_blocks = 2, dropout_rate = 0.2):
        super().__init__()
        self.context_encoder = ContextMLP_withMaps(num_mri_raw,context_size,dropout_rate)
        self.num_mri_raw = num_mri_raw
        self.num_param_maps = num_channels-num_mri_raw
        #K = 3
        latent_size = 2
        #hidden_units = 128
        #num_blocks = 2
        self.total_context = context_size+self.num_param_maps
        flows = []
        for i in range(K):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units, 
                                                          context_features=self.total_context, 
                                                          num_blocks=num_blocks)]
            flows += [nf.flows.LULinearPermute(latent_size)]
       
        q0 = nf.distributions.DiagGaussian(2, trainable=False)
        nfm = nf.ConditionalNormalizingFlow(q0, flows)
        self.flow_model = nfm

        
    def forward(self, x, task, samples=None, num_samples=None):
        context = self.context_encoder(x, self.num_mri_raw)
        if task==0:  # Flow model
            # trick to use all samples at once
            context_rep = torch.repeat_interleave(context, num_samples, 0)
            l = self.flow_model.forward_kld(samples, context_rep)
            return l
        elif task==1: #Classification task
            return self.region_output(context), self.group_output(context)
        elif task==2: # only region
            return self.region_output(context)
        elif task==3: # only group
            return self.group_output(context)