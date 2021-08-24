from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss

class LearnedLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        coarse_loss = self.loss(inputs['rgb_coarse'], targets).mean(-1)
        module_loss = LossPredLoss(inputs['rgb_loss_coarse'], coarse_loss)
        rgb_loss = coarse_loss.mean()

        if 'rgb_fine' in inputs:
            fine_loss = self.loss(inputs['rgb_fine'], targets).mean(-1)
            module_loss += self.coef * LossPredLoss(inputs['rgb_loss_fine'], fine_loss)
            rgb_loss += self.coef * fine_loss.mean()

        return rgb_loss, module_loss


class UncertainLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        coarse_loss = self.loss(inputs['rgb_coarse'], targets).mean(-1)
        beta_val = inputs['rgb_beta_coarse']
        rgb_loss = coarse_loss.mean() / (2 *  (beta_val**2).mean())
        rgb_loss += torch.mean(0.5 * (torch.log(beta_val)**2))

        if 'rgb_fine' in inputs:
            fine_loss = self.loss(inputs['rgb_fine'], targets).mean(-1)
            beta_val = inputs['rgb_beta_fine'] ** 2
            rgb_loss += self.coef * fine_loss.mean() / (2 *  (beta_val**2).mean())
            rgb_loss += torch.mean(0.5 * (torch.log(beta_val)**2))

        return rgb_loss

class EvidentialLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.lam = 2.0
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        alpha, beta, gamma, v = inputs['rgb_alpha_coarse'], inputs['rgb_beta_coarse'], inputs['rgb_coarse'], inputs['rgb_nu_coarse']
        rgb_loss = NIG_NLL(targets, gamma, v, alpha, beta) + self.lam*NIG_Reg(targets, gamma, v, alpha, beta)

        if 'rgb_fine' in inputs:
            alpha, beta, gamma, v = inputs['rgb_alpha_fine'], inputs['rgb_beta_fine'], inputs['rgb_fine'], inputs['rgb_nu_fine']
            rgb_loss += NIG_NLL(targets, gamma, v, alpha, beta) + self.lam*NIG_Reg(targets, gamma, v, alpha, beta)

        return rgb_loss

def NIG_NLL(y_true, gamma, v, alpha, beta):
    twoBLambda = 2 * beta * (1 + v)

    nll = 0.5 * torch.log(np.pi/v) \
        - alpha*torch.log(twoBLambda) \
        + (alpha + 0.5) * torch.log(v * (y_true - gamma)**2 + twoBLambda) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    
    return torch.mean(nll)

def NIG_Reg(y_true, gamma, v, alpha, beta):
    error = torch.abs(y_true - gamma)
    evi = 2*v + alpha
    reg = error * evi

    return torch.mean(reg)


def LossPredLoss(input, target, margin=0.001, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    target = target.detach()

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * (input), min=0)) + torch.sum(F.mse_loss(input, target, size_average='none'))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * (input), min=0)
    else:
        NotImplementedError()
    
    return loss


def LossPredLossScaled(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    target = target.detach()

    mean_input = input.mean()
    mean_target = target.mean()

    input = input / input.mean()
    target = target / target.mean()

    error_term = (target - input)**2 #(mean_target - mean_input)
    
    if reduction == 'mean':
        loss = torch.mean(error_term)
    elif reduction == 'none':
        loss = error_term
    else:
        NotImplementedError()
    
    return loss

loss_dict = {
            'color': ColorLoss, 
            'llal': LearnedLoss, 
            'uncertainty': UncertainLoss,
            'evidential': EvidentialLoss
            }
