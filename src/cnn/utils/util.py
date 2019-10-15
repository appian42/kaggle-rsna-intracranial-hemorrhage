import os
import random
import glob

import pandas as pd
import numpy as np
import torch

from .logger import log


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lr(optim):
    if optim:
        return optim.param_groups[0]['lr']
    else:
        return 0


def save_model(model, optim, detail, fold, dirname):
    path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'detail': detail,
    }, path)
    log('saved model to %s' % path)


def load_model(path, model, optim=None):

    # remap everthing onto CPU 
    state = torch.load(str(path), map_location=lambda storage, location: storage)

    model.load_state_dict(state['model'])
    if optim:
        log('loading optim too')
        optim.load_state_dict(state['optim'])
    else:
        log('not loading optim')

    model.cuda()

    detail = state['detail']
    log('loaded model from %s' % path)

    return detail
