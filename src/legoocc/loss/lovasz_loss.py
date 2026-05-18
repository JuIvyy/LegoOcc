from .base_loss import BaseLoss
from . import GPD_LOSS
from legoocc.occutils.lovasz_losses import lovasz_softmax, lovasz_hinge, global_lovasz_softmax
import torch


@GPD_LOSS.register_module()
class LovaszLoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=None, input_dict=None, activated=False, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
        self.ignore_label = ignore_label
        self.activated = activated
    
    def lovasz_loss(self, lovasz_input, lovasz_label, fov_mask):
        # input: -1, c, h, w, z
        # output: -1, h, w, z
        if not self.activated:
            lovasz_input = torch.softmax(lovasz_input.float(), dim=1)
        lovasz_label = lovasz_label.long()
        lovasz_loss = lovasz_softmax(lovasz_input, lovasz_label, ignore=self.ignore_label, fov_mask=fov_mask)
        return lovasz_loss

@GPD_LOSS.register_module()
class GlobalLovaszLoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
        self.ignore_label = ignore_label
    
    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, c, h, w, z
        # output: -1, h, w, z
        lovasz_input = torch.softmax(lovasz_input.float(), dim=1)
        lovasz_label = lovasz_label.long()
        lovasz_loss = global_lovasz_softmax(lovasz_input, lovasz_label, ignore=self.ignore_label)
        return lovasz_loss


@GPD_LOSS.register_module()
class LovaszHingeLoss(BaseLoss):
    
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
    
    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, h, w, z
        # output: -1, h, w, z
        lovasz_input = lovasz_input.float()
        lovasz_label = lovasz_label.long()
        lovasz_loss = lovasz_hinge(lovasz_input, lovasz_label)
        return lovasz_loss
