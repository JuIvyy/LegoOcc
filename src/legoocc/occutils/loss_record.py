import numpy as np

class LossRecord():
    
    def __init__(self, loss_func, num_layers=1) -> None:
        self.loss_dict = dict()
        if loss_func:
            for loss in loss_func.losses:
                loss_name = getattr(loss, 'loss_name', loss.__class__.__name__)
                # loss_name = loss.loss_name
                self.loss_dict[f'{loss_name}'] = []
                # for lid in range(num_layers):
                #     self.loss_dict[f'{loss_name}_l{lid}'] = []
        self.total_loss = []
    
    def reset(self):
        for key in self.loss_dict.keys():
            self.loss_dict[key] = []
        self.total_loss = []

    def update(self, loss, loss_dict):
        for key in loss_dict.keys():
            if key in self.loss_dict:
                self.loss_dict[key].append(loss_dict[key])
            else:
                self.loss_dict[key] = [loss_dict[key]]
        self.total_loss.append(loss)
    
    def loss_info(self):
        info = ''
        for name, loss_list in self.loss_dict.items():
            if len(loss_list):
                info += '%s: %.3f (%.3f),   ' % (name, loss_list[-1], np.mean(loss_list))
        info += 'Loss: %.3f (%.3f),   ' % (self.total_loss[-1], np.mean(self.total_loss))
        
        return info