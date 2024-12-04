import math
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .model.model import Model
from .loss_function import  kl_loss, penalty_loss, entropy_loss, L1_reg
from .utils import select_device, set_random_seed, factor



class Integrator():
    def __init__(self, input_dict,init_alpha=1.0,lamda_batch=None, cond_factor_k=0.5, device = 'cuda:0'):

        self.device = select_device(device)

        set_random_seed()

        self.lamda_batch = lamda_batch
        self.cond_factor_k = cond_factor_k

        self.exp_data = input_dict["exp"]
        self.num_features = input_dict["num_features"]
        self.num_batches = input_dict["num_batches"]
        self.num_conditions = input_dict["num_conditions"]

        self.batch_labels = input_dict["batch_labels"]
        self.condition_labels = input_dict["condition_labels"]
        self.cond_ratio = input_dict["cond_ratio"]
        self.batch_weights = input_dict["batch_weights"]
        self.max_batch = input_dict["max_batch"]

        self.model = Model(self.num_features, self.num_batches, self.num_conditions,init_alpha).to(self.device)


    def fit(self, batch_size=300, lr=None, mu0=0.001,\
            num_epochs=100, glr_factor=1.0, M = 10, \
                entropy_weight = 0.01, print_freq=5, kl_scale = 1e-4,**kws):
        
        
        x = torch.Tensor(self.exp_data)
        b = torch.LongTensor(self.batch_labels)
        c = torch.LongTensor(self.condition_labels)
        weights = torch.FloatTensor(self.batch_weights)
        cond_Error = torch.FloatTensor(self.cond_ratio)
        
        if torch.cuda.is_available():
            weights = weights.to(self.device)

            
        tqdm.write('Starting training...')
        
        
        dataset = TensorDataset(x, b, c,cond_Error)
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True,
                                num_workers=0,drop_last=False)
        
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-6)
        
        ae_criterion = nn.MSELoss()

        batch_criterion = nn.NLLLoss(weight=weights,reduction='mean')

        condition_criterion = nn.NLLLoss(reduction='none')
        train_losses = []

        progress = tqdm(range(num_epochs+1), ncols=100, 
                        desc="Epoch Progress",  dynamic_ncols=True, 
                        bar_format="{l_bar}{bar}| [{elapsed}<{remaining},{rate_fmt}{postfix}]" ,leave=True)
        
        for epoch in progress:
            
            epoch_total_loss = 0
            if glr_factor:
                alpha = glr_factor
            else:
                alpha=factor(epoch)
                

            if self.lamda_batch:
                self.lamda_batch = self.lamda_batch           
            else:
                self.lamda_batch = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1     
            
            if lr:
                current_lr = lr
            else:
                mu0, alpha_lr, beta = mu0, 10, 0.75
                current_lr = mu0 / (1 + alpha_lr * (epoch) / num_epochs) ** beta
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


            for i, (batch_x, batch_b, batch_c, batch_cond_Error) in enumerate(data_loader):

                batch_x = batch_x.to(self.device)
                batch_b = batch_b.to(self.device)
                batch_c = batch_c.to(self.device)
                batch_cond_Error = batch_cond_Error.to(self.device)


                batch_output, cond_output, decode_output, z, mean, log_var, Zu, Zc = self.model.forward(batch_x, batch_b, alpha)

                kl_z_loss = kl_loss(mean, log_var)
                
                ae_loss = ae_criterion(batch_x, decode_output)
                
                batch_loss = batch_criterion(torch.log(batch_output), batch_b)
                batch_loss = batch_loss / np.log(self.num_batches)
                
                cond_loss = condition_criterion(torch.log(cond_output), batch_c)
                entropy_cond_loss = entropy_loss(cond_loss)
                
                cond_loss, cond_loss_rej = penalty_loss(cond_loss, self.num_conditions, batch_cond_Error, M, self.cond_factor_k)

                total_loss = 2 * ae_loss + self.lamda_batch * batch_loss + cond_loss_rej + kl_scale*kl_z_loss + 1e-3 * L1_reg(self.model.condition_classifier) + entropy_weight * entropy_cond_loss
 
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                    
                epoch_total_loss += total_loss.item()
                avg_epoch_loss = epoch_total_loss / len(data_loader)
            train_losses.append(avg_epoch_loss)
            

            progress.set_postfix_str(f"epoch: {epoch}/{num_epochs}, total_loss={avg_epoch_loss:.2f}, current_lr={current_lr:.6f}")

                 
        tqdm.write('Training has been finished.')
        
        plt.plot(range(len(train_losses)), train_losses,label="",linewidth=1.5,color='b')
        plt.xlabel("Iteration (epoch)")  
        plt.ylabel("Loss")
        plt.title('Total Loss per epoch')
        plt.show()
        


    
        
    def get_corrected_exp(self, input_dict, **kws):
        
        
        set_random_seed()
        b_1 = np.ones(input_dict["exp"].shape[0],dtype="int8")

        Z, Zu, Zc, x_0, pc = self.transform(
            exp_data=input_dict['exp'], 
            batch_labels=input_dict["batch_labels"], 
            decode_batch_labels=b_1 * input_dict["max_batch"],
            device=self.device
            
        )
        

        df = pd.DataFrame(x_0)
        

        p = pc[np.arange(pc.shape[0]), input_dict["condition_labels"]]
        rej = np.where((1 - p) < self.cond_factor_k * (input_dict["num_conditions"] - 1) / input_dict["num_conditions"], 'Sensitive', 'Non-sensitive')
        rej = rej.reshape(input_dict["exp"].shape[0], 1)

        adata1 = sc.AnnData(x_0)
        adata1.obs = input_dict["meta"].copy() 

        adata1.obs['condition_output'] = p.tolist()
        adata1.obs["sensitive"] = rej
        

        adata1.obsm["Z_emb"] = Z
        adata1.obsm["Zu_emb"] = Zu
        adata1.obsm["Zc_emb"] = Zc

        return adata1
        
    

    def transform(self, exp_data, batch_labels, batch_size=None, decode_batch_labels=None, device='cuda:0', **kws):
        if type(exp_data) is np.ndarray:
            x = torch.Tensor(exp_data)
            b = torch.LongTensor(batch_labels)
            if decode_batch_labels is not None:
                b_ = torch.LongTensor(decode_batch_labels)
                
        else:
            x = exp_data
            b = batch_labels
            if decode_batch_labels is not None:
                b_ = decode_batch_labels
            

                
        if batch_size is None:
            batch_size = x.shape[0]
        
        if decode_batch_labels is None:
            b_ = b
                
        dataset = TensorDataset(x, b, b_)
        data_loader = DataLoader(dataset, 
                                batch_size=batch_size, 
                                shuffle=False,
                                num_workers=0)


        with torch.no_grad():
            encode_output, decode_output, cond_output = [], [], []
            Zu_output,Zc_output = [], []
            for batch in data_loader:
                batch_x, batch_b, batch_b_ = batch
                
                
                batch_x = batch_x.to(device)
                batch_b = batch_b.to(device)
                batch_b_ = batch_b_.to(device)
                
                batch_b = F.one_hot(batch_b, num_classes=self.num_batches)
                batch_b_ = F.one_hot(batch_b_, num_classes=self.num_batches)
                batch_encode_output, _, _, batch_Zu,batch_Zc = self.model._encode(batch_x, batch_b)
                batch_decode_output = self.model._decode(batch_encode_output, batch_b_)

                
                batch_cond_output = self.model.condition_classifier(batch_Zc)
                batch_cond_output = torch.softmax(batch_cond_output, dim=1)
                
                encode_output.append(batch_encode_output)
                Zu_output.append(batch_Zu)
                Zc_output.append(batch_Zc)
                decode_output.append(batch_decode_output)
                cond_output.append(batch_cond_output)
                
            encode_output = torch.cat(encode_output, dim=0).cpu().detach().numpy()
            Zu_output = torch.cat(Zu_output, dim=0).cpu().detach().numpy()
            Zc_output = torch.cat(Zc_output, dim=0).cpu().detach().numpy()
            decode_output = torch.cat(decode_output, dim=0).cpu().detach().numpy()
            cond_output = torch.cat(cond_output, dim=0).cpu().detach().numpy()
        
        return encode_output, Zu_output, Zc_output, decode_output, cond_output
    
    def fit_transform(self, **kws):
        self.fit(**kws)
        return self.transform(**kws)