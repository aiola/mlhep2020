import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class LinearSVDO(nn.Module):
    def __init__(self, in_features, out_features, threshold, device, bias=True):
        super(LinearSVDO, self).__init__()
        """
            in_features: int, a number of input features
            out_features: int, a number of neurons
            threshold: float, a threshold for clipping weights
        """
        
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.device = device

        self.mu = Parameter(torch.Tensor(self.out_features, self.in_features))
        # torch.nn.parameter.Parameter of size out_features x in_features
        self.log_sigma = Parameter(torch.Tensor(self.out_features, self.in_features))
        # torch.nn.parameter.Parameter of size out_features x in_features
        self.bias = Parameter(torch.Tensor(1, self.out_features))
        # torch.nn.parameter.Parameter of size 1 x out_features
        self.reset_parameters()
        self.k1, self.k2, self.k3 = torch.Tensor([0.63576]).to(device), torch.Tensor([1.8732]).to(device), torch.Tensor([1.48695]).to(device)
        
    def reset_parameters(self):
        self.bias.data.zero_()
        self.mu.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):      
        # x is a torch.Tensor of shape (?number_of_objects, in_features)
        # log_alpha is a torch.Tensor of shape (out_features, in_features)
        self.log_alpha = 2 * self.log_sigma - self.mu.pow(2).log() # Compute using self.log_sigma and self.mu
        # clipping for a numerical stability
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)   
        
        if self.training:
            # lrt_mean is a torch.Tensor of shape (x.shape[0], out_features)
            lrt_mean = x @ self.mu.t()
            # compute mean activation using LRT
            # lrt_std is a torch.Tensor of shape (x.shape[0], out_features)
            lrt_std = (1e-8 + x.pow(2) @ (self.log_alpha.exp() * self.mu.pow(2)).t()).sqrt()
            # compute std of activations unsig lrt, 
            # do not forget use torch.sqrt(x + 1e-8) instead of torch.sqrt(x)
            # eps is a torch.Tensor of shape (x.shape[0], out_features)
            eps = torch.randn(x.shape[0], self.out_features).to(self.device)
            # sample of noise for reparametrization
            return lrt_mean + lrt_std * eps  + self.bias
            # sample of activation
        
        mu = self.mu * torch.le(self.log_alpha, self.threshold).float()
        out = x @ mu.t() + self.bias
        
        # compute the output of the layer
        # use weighs W = Eq = self.mu
        # clip all weight with log_alpha > threshold
        return out
    
    
    def count_parameters(self):
        self.log_alpha = 2 * self.log_sigma - self.mu.pow(2).log()
        total = self.log_alpha.numel() + self.bias.numel()
        effective = torch.le(self.log_alpha, self.threshold).sum().item()
        effective += self.bias.numel()
        return (effective, total)
        
        
    def kl_reg(self):
        self.log_alpha = 2 * self.log_sigma - self.mu.pow(2).log() # Compute using self.log_sigma and self.mu
        # clipping for a numerical stability
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10)
        
        kl = (self.k1 * (self.k2 + self.k3 * self.log_alpha).sigmoid() - 
              0.5 * (-self.log_alpha).exp().log1p()).sum()
        # eval KL using the approximation
        return kl

class SGVLB(nn.Module):
    def __init__(self, net, train_size):
        super(SGVLB, self).__init__()
        self.train_size = train_size # int, the len of dataset
        self.net = net # nn.Module
        self.nllloss  = torch.nn.NLLLoss()
        
    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
                kl = kl + module.kl_reg()
                
        sgvlb_loss = - kl_weight * kl + self.train_size * self.nllloss(input, target) 
        # a scalar torch.Tensor, SGVLB loss
        return sgvlb_loss