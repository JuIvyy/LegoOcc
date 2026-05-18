from .base_loss import BaseLoss
from . import GPD_LOSS


@GPD_LOSS.register_module()
class L1Loss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'flow': 'flow',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.l1_loss
        
    def l1_loss(self, flow):
        flow = flow.float()
            
        loss = flow.abs()
        
        return loss.mean()