import lookahead
import mtt

import torch.optim as optim

"""
Usage:
    get_opt_factory:
        return optimizer, <_opt_description>
"""
def get_lookahead_factory(base_opt_fac, **kwargs):
    return lambda pa: lookahead.Lookahead(base_opt_fac(pa), **kwargs)
def get_mtt_factory(base_opt_fac_s, **kwargs):
    print('mtt:', kwargs['v_s'], kwargs['k'])
    return lambda pa: mtt.Multipath([opt_fac(pa) for opt_fac in base_opt_fac_s], **kwargs)

def get_base_optimizer_factory(opt_name, **kwargs):
    print('base:%s' % opt_name)
    if opt_name == 'sgd':
        return lambda pa: optim.SGD(pa, momentum = 0, **kwargs)
    if opt_name == 'mom':
        return lambda pa: optim.SGD(pa, momentum = 0.9, **kwargs)
    if opt_name == 'mom2':
        return lambda pa: optim.SGD(pa, momentum = 0.99, **kwargs)
    if opt_name == 'mom3':
        return lambda pa: optim.SGD(pa, momentum = 0.999, **kwargs)

