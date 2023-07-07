from typing import Dict
from torch import optim
import torch


def get_optimizer(params, optimizer_config: Dict = None):
    """ wrapper to return appropriate optimizer class """
    opt_alg = optimizer_config.get('optimizer', 'SGD')
    if optimizer_config is None:
        optimizer_config = {}
    if opt_alg == 'SGD':
        return optim.SGD(params=params,
                         lr=optimizer_config.get('lr0', 0.1),
                         momentum=optimizer_config.get('momentum', 0.9),
                         weight_decay=optimizer_config.get('reg', 0.005),
                         nesterov=optimizer_config.get('nesterov', True))

    elif opt_alg == 'Adam':
        return optim.Adam(params=params,
                          lr=optimizer_config.get('lr0', 1),
                          betas=optimizer_config.get('betas', (0.5, 0.99)),
                          eps=optimizer_config.get('eps', 1e-08),
                          weight_decay=optimizer_config.get('reg', 0.05),
                          amsgrad=optimizer_config.get('amsgrad', False))
    elif opt_alg == 'AdamW':
        return optim.AdamW(params=params,
                           lr=optimizer_config.get('lr0', 1),
                           betas=optimizer_config.get('betas', (0.9, 0.999)),
                           eps=optimizer_config.get('eps', 1e-08),
                           weight_decay=optimizer_config.get('reg', 0.05),
                           amsgrad=optimizer_config.get('amsgrad', False))

    elif opt_alg == 'lars':
        return LARS(params=params,
                    lr=optimizer_config.get('lr0', 1),
                    momentum=optimizer_config.get('momentum', 0.9),
                    weight_decay=optimizer_config.get('reg', 0),
                    eta=0.001,
                    weight_decay_filter=None,
                    lars_adaptation_filter=None)

    else:
        raise NotImplementedError


def get_scheduler(optimizer, lrs_config: Dict = None):
    lrs = lrs_config.get('lrs')
    if lrs == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer,
                                         step_size=lrs_config.get('step_size'),
                                         gamma=lrs_config.get('gamma'))
    elif lrs == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                    T_max=lrs_config.get('T_max'),
                                                    eta_min=lrs_config.get('eta_min', 0))
    elif lrs == 'cosine_warm':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                              T_0=lrs_config.get('T0'),
                                                              T_mult=lrs_config.get('T_mult', 1),
                                                              eta_min=lrs_config.get('eta_min', 0))
    elif lrs == 'multi_step':
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                              milestones=lrs_config.get('milestones'),
                                              gamma=lrs_config.get('gamma'))
    else:
        return None


# ----- Optimizer Implementations ----
class LARS(optim.Optimizer):
    """ Large Batch Training of Convolutional Networks : https://arxiv.org/abs/1708.03888 """

    def __init__(self,
                 params,
                 lr,
                 weight_decay=0,
                 momentum=0.9,
                 eta=0.001,
                 weight_decay_filter=None,
                 lars_adaptation_filter=None):

        defaults = dict(lr=lr,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        eta=eta,
                        weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super(LARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        """ Performs a single optimization step """
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
