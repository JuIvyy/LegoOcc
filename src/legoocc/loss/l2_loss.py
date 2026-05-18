from .base_loss import BaseLoss
from . import GPD_LOSS


@GPD_LOSS.register_module()
class L2Loss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'bev_pred': 'bev_pred',
                'bev_gt': 'bev_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.l2_loss
        
    def l2_loss(self, bev_pred, bev_gt):
        bev_pred = bev_pred.float()
        bev_gt = bev_gt.float()
            
        loss = (bev_pred - bev_gt) ** 2
        
        return loss.mean()