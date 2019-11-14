import sys
import os
import time
import argparse
import random
import collections
import pickle

from apex import amp
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss

import torch
from torch import nn
import torch.nn.functional as F

from . import factory
from .utils import util
from .utils.config import Config
from .utils.logger import logger, log


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'valid', 'test'])
    parser.add_argument('config')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--snapshot')
    parser.add_argument('--output') 
    parser.add_argument('--n-tta', default=1, type=int)
    return parser.parse_args()


def main():

    args = get_args()
    cfg = Config.fromfile(args.config)

    # copy command line args to cfg
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.fold = args.fold
    cfg.snapshot = args.snapshot
    cfg.output = args.output
    cfg.n_tta = args.n_tta
    cfg.gpu = args.gpu

    logger.setup(cfg.workdir, name='%s_fold%d' % (cfg.mode, cfg.fold))
    torch.cuda.set_device(cfg.gpu)
    util.set_seed(cfg.seed)

    log(f'mode: {cfg.mode}')
    log(f'workdir: {cfg.workdir}')
    log(f'fold: {cfg.fold}')
    log(f'batch size: {cfg.batch_size}')
    log(f'acc: {cfg.data.train.n_grad_acc}')

    model = factory.get_model(cfg)
    model.cuda()

    if cfg.mode == 'train':
        train(cfg, model)
    elif cfg.mode == 'valid':
        valid(cfg, model)
    elif cfg.mode == 'test':
        test(cfg, model)


def test(cfg, model):
    assert cfg.output
    util.load_model(cfg.snapshot, model)
    loader_test = factory.get_dataloader(cfg.data.test)
    with torch.no_grad():
        results = [run_nn(cfg.data.test, 'test', model, loader_test) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)


def valid(cfg, model):
    assert cfg.output
    criterion = factory.get_loss(cfg)
    util.load_model(cfg.snapshot, model)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold] if cfg.fold is not None else None)
    with torch.no_grad():
        results = [run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion) for i in range(cfg.n_tta)]
    with open(cfg.output, 'wb') as f:
        pickle.dump(results, f)
    log('saved to %s' % cfg.output)


def train(cfg, model):

    criterion = factory.get_loss(cfg)
    optim = factory.get_optim(cfg, model.parameters())

    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }
    if cfg.resume_from:
        detail = util.load_model(cfg.resume_from, model, optim=optim)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })

    folds = [fold for fold in range(cfg.n_fold) if cfg.fold != fold]
    loader_train = factory.get_dataloader(cfg.data.train, folds)
    loader_valid = factory.get_dataloader(cfg.data.valid, [cfg.fold])

    log('train data: loaded %d records' % len(loader_train.dataset))
    log('valid data: loaded %d records' % len(loader_valid.dataset))

    scheduler = factory.get_scheduler(cfg, optim, best['epoch'])

    log('apex %s' % cfg.apex)
    if cfg.apex:
        amp.initialize(model, optim, opt_level='O1')

    for epoch in range(best['epoch']+1, cfg.epoch):

        log(f'\n----- epoch {epoch} -----')

        #util.set_seed(epoch)

        run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optim=optim, apex=cfg.apex)

        #util.set_seed(1000)
        with torch.no_grad():
            val = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion)

        detail = {
            'score': val['score'],
            'loss': val['loss'],
            'epoch': epoch,
        }
        if val['loss'] <= best['loss']:
            best.update(detail)

        util.save_model(model, optim, detail, cfg.fold, cfg.workdir)
            
        log('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
            
        #scheduler.step(val['loss']) # reducelronplateau
        scheduler.step()


def run_nn(cfg, mode, model, loader, criterion=None, optim=None, scheduler=None, apex=None):
    if mode in ['train']:
        model.train()
    elif mode in ['valid', 'test']:
        model.eval()
    else:
        raise RuntimeError('Unexpected mode %s' % mode)

    t1 = time.time()
    losses = []
    ids_all = []
    targets_all = []
    outputs_all = []

    for i, (inputs, targets, ids) in enumerate(loader):

        batch_size = len(inputs)

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)

        if mode in ['train', 'valid']:
            loss = criterion(outputs, targets)
            with torch.no_grad():
                losses.append(loss.item())

        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optim) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() # accumulate loss
            if (i+1) % cfg.n_grad_acc == 0:
                optim.step() # update
                optim.zero_grad() # flush
            
        with torch.no_grad():
            ids_all.extend(ids)
            targets_all.extend(targets.cpu().numpy())
            outputs_all.extend(torch.sigmoid(outputs).cpu().numpy())
            #outputs_all.append(torch.softmax(outputs, dim=1).cpu().numpy())

        elapsed = int(time.time() - t1)
        eta = int(elapsed / (i+1) * (len(loader)-(i+1)))
        progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} loss200:{(np.sum(losses[-200:])/(min(i+1,200))):.6f} lr:{util.get_lr(optim):.2e}'
        print(progress, end='')
        sys.stdout.flush()

    result = {
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
        'loss': np.sum(losses) / (i+1),
    }

    if mode in ['train', 'valid']:
        result.update(calc_auc(result['targets'], result['outputs']))
        result.update(calc_logloss(result['targets'], result['outputs']))
        result['score'] = result['logloss']

        log(progress + ' auc:%.4f micro:%.4f macro:%.4f' % (result['auc'], result['auc_micro'], result['auc_macro']))
        log('%.6f %s' % (result['logloss'], np.round(result['logloss_classes'], 6)))
    else:
        log('')

    return result


def calc_logloss(targets, outputs, eps=1e-5):
    # for RSNA
    try:
        logloss_classes = [log_loss(np.round(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]
    except ValueError as e: 
        logloss_classes = [1, 1, 1, 1, 1, 1]

    return {
        'logloss_classes': logloss_classes,
        'logloss': np.average(logloss_classes, weights=[2,1,1,1,1,1]),
    }


def calc_auc(targets, outputs):
    macro = roc_auc_score(np.round(targets), outputs, average='macro')
    micro = roc_auc_score(np.round(targets), outputs, average='micro')
    return {
        'auc': (macro + micro) / 2,
        'auc_macro': macro,
        'auc_micro': micro,
    }



if __name__ == '__main__':

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('benchmark', torch.backends.cudnn.benchmark)

    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')
