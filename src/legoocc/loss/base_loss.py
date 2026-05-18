import torch.nn as nn
import torch


class BaseLoss(nn.Module):

    """ Base loss class.
    args:
        weight: weight of current loss.
        input_keys: keys for actual inputs to calculate_loss().
            Since "inputs" may contain many different fields, we use input_keys
            to distinguish them.
        loss_func: the actual loss func to calculate loss.
    """

    def __init__(
            self, 
            weight=1.0,
            input_dict={'input': 'input'},
            **kwargs):
        super().__init__()
        self.weight = weight
        self.input_dict = input_dict
        self.loss_func = lambda: 0

    def forward(self, inputs, force_reduction_none=False):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs.update({input_key: inputs[input_val]})

        if force_reduction_none:
            actual_inputs['reduction'] = 'none'

        loss = self.loss_func(**actual_inputs)

        loss = torch.nan_to_num(loss)

        return self.weight * loss
