import torch
import torch.nn as nn
import torch.nn.functional as F
from .GRL import GradientReversal




class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class Model(nn.Module):
    def __init__(self, num_features, num_batches, num_conditions,init_alpha=1.0):
        super(Model, self).__init__()

        self.num_features = num_features
        self.num_batches = num_batches
        self.num_conditions = num_conditions
        self.alpha = init_alpha

        emd_dim = 32
        self.encoder = nn.Sequential(
            nn.Linear(num_features + num_batches, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, emd_dim * 2)  
        )

        self.decoder = nn.Sequential(
            nn.Linear(emd_dim + num_batches, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, num_features),
            nn.Tanh()
        )

        self.dec_mean = nn.Sequential(
            nn.Linear(128, num_features), 
            MeanAct()
        )

        self.dec_disp = nn.Sequential(
            nn.Linear(128, num_features), 
            DispAct()
        )

        self.dec_pi = nn.Sequential(
            nn.Linear(128, num_features), 
            nn.Sigmoid()
        )


        self.condition_classifier = nn.Sequential(
            nn.Linear(8, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, num_conditions)
        )

        self.batch_classifier = nn.Sequential(
            GradientReversal(alpha=self.alpha),
            nn.Linear(emd_dim, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, num_batches)
        )

        if torch.cuda.is_available():
            print('cuda is available.')


    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  
        eps = torch.randn_like(std)  
        z = mean + eps * std 
        return z
    
    def _encode(self, x, one_hot_b, apply_one_hot=False):
        if apply_one_hot:
            one_hot_b = F.one_hot(one_hot_b, num_classes=self.num_batches)

        z_params = self.encoder(torch.cat((x, one_hot_b), dim=1))
        mean, log_var = z_params[:, :32], z_params[:, 32:]  

        z = self.reparameterize(mean, log_var)

        # z = F.normalize(z, dim=1)
        Zu, Zc = z[:, :24], z[:, 24:]

        return z, mean, log_var, Zu, Zc

    def _decode(self, z, one_hot_b):

        return self.decoder(torch.cat((z, one_hot_b), dim=1))

    def forward(self, x, b, alpha):
        # self.batch_classifier[0].set_factor(alpha)

        b = F.one_hot(b, num_classes=self.num_batches)
        z, mean, log_var, Zu, Zc = self._encode(x, b)
        
        decode_output = self._decode(z, b)
        decode_output = 1 * decode_output
        
        self.alpha=alpha

        batch_output = self.batch_classifier(z)
        batch_output = torch.softmax(batch_output, dim=1)

        cond_output = self.condition_classifier(Zc)
        cond_output = torch.softmax(cond_output, dim=1)

        return batch_output, cond_output, decode_output, z, mean, log_var, Zu, Zc