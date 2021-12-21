# -- coding: utf-8 --
import torch.nn.init as init
import math


def weights_init(init_type='gaussian'):
    def init_fun(model):
        classname = model.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(model.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(model.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(model.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)

            if hasattr(model, 'bias') and model.bias is not None:
                init.constant_(model.bias.data, 0.0)

    return init_fun

