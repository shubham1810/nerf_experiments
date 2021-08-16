from torch import nn
import torch

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


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

loss_dict = {'color': ColorLoss, 'llal': LearnedLoss}