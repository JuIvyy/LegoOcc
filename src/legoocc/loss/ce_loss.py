from .base_loss import BaseLoss
from . import GPD_LOSS
import torch.nn.functional as F
import torch


@GPD_LOSS.register_module()
class CELoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=255, loss_name=None,
                 cls_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        if loss_name is not None:
            self.loss_name = loss_name
        self.loss_func = self.ce_loss
        self.ignore_label = ignore_label
        self.cls_weight = torch.tensor(cls_weight) if cls_weight is not None else None
    
    def ce_loss(self, ce_input, ce_label, fov_mask):
        # input: -1, c
        # output: -1, 1
        ce_input = ce_input.float().permute(2, 3, 4, 0, 1)
        ce_label = ce_label.long().permute(1, 2, 3, 0)
        ce_input = ce_input[fov_mask].permute(1, 2, 0)
        ce_label = ce_label[fov_mask].permute(1, 0)
        weight = self.cls_weight.to(ce_input.device)
        
        ce_loss = F.cross_entropy(ce_input, ce_label, weight=weight, 
                                  ignore_index=self.ignore_label)
        return ce_loss


@GPD_LOSS.register_module()
class BCELoss(BaseLoss):
    
    def __init__(self, weight=1.0, pos_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
    
    def ce_loss(self, ce_input, ce_label, fov_mask):
        # input: -1, 1
        # output: -1, 1
        n_relations, _, _ = ce_input.shape
        ce_input = ce_input.float().reshape(n_relations, -1).T
        ce_label = ce_label.float().reshape(n_relations, -1).T # M, 4
        
        cnt_neg = (ce_label == 0).sum(0)
        cnt_pos = ce_label.sum(0)
        pos_weight = cnt_neg / cnt_pos
        ce_loss = F.binary_cross_entropy_with_logits(ce_input, ce_label, weight=pos_weight)
        return ce_loss