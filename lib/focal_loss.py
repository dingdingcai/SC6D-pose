import torch
from torch  import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, alpha=0.5, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        assert(isinstance(gamma, (float, int)))
        assert(isinstance(alpha, (float, int)) and alpha > 0)
        
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        assert(len(inputs)== len(targets) and inputs.dim() <= 3)
        
        if inputs.dim() == 3:
            N, C, Q = inputs.shape
            if targets.dim() == 1: # N
                targets = F.one_hot(targets, num_classes=C).view(N, C, 1).repeat(1, 1, Q) # NxCxQ  
            elif targets.dim() == 2: # NxQ
                targets = F.one_hot(targets, num_classes=C).permute(0, 2, 1) # NxCxQ
            elif targets.dim() == 3: # NxCxQ
                assert(inputs.shape == targets.shape)
            else:
                raise('target dimension no greater than inputs')
        else:
            N, C = inputs.shape
            if targets.dim() == 1: # N
                targets = F.one_hot(targets, num_classes=C) # NxC
            elif targets.dim() == 2: # NxQ
                assert(inputs.shape == targets.shape)
            else:
                raise('target dimension no greater than inputs')
            inputs = inputs.view(N, C, 1)
            targets = targets.view(N, C, 1)
        
        Pr = (F.softmax(inputs, dim=1) * targets).sum(dim=1).float()         # BxCxQ->BxQ
        Pr_log = (F.log_softmax(inputs, dim=1) * targets).sum(dim=1).float() # BxCxQ->BxQ
        batch_loss = -self.alpha * (1.0 - Pr) ** self.gamma * Pr_log
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss