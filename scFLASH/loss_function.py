import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch


def L1_reg(model):
    reg = 0
    for param in model.parameters():
        reg += torch.sum(torch.abs(param))
    return reg

def kl_loss(mean, log_var):
    kl_z_loss = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
    kl_z_loss = torch.sum(kl_z_loss,dim=1)
    kl_z_loss = torch.mean(kl_z_loss)
    return kl_z_loss

def penalty_loss(cond_loss, num_conditions, batch_cond_Error, M, reject_cost):
               
    cond_loss = 1 - torch.exp(-cond_loss)
    cond_loss = torch.dot(cond_loss,batch_cond_Error)/torch.sum(batch_cond_Error)
    cond_loss = cond_loss / ((num_conditions - 1) / num_conditions) 
               
    cost = reject_cost
    cond_loss_rej = M * torch.relu(cond_loss - cost)
    
    return cond_loss, cond_loss_rej
                               
def entropy_loss(cond_loss):                        
    prop_cond_loss = cond_loss / (torch.sum(cond_loss) + 1e-6)
    entropy_cond_loss = -torch.sum(prop_cond_loss * torch.log(prop_cond_loss + 1e-6)) / np.log(prop_cond_loss.shape[0])
    return entropy_cond_loss
                
def L1_reg(model):
    reg = 0
    for param in model.parameters():
        reg += torch.sum(torch.abs(param))
    return reg













