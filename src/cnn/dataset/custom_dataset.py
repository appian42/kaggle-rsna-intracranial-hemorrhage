import os
import pickle
import random

import pandas as pd
import numpy as np
np.seterr(over='ignore')
import torch
import cv2
import pydicom

from .. import factory
from ..utils.logger import log
from ...utils import mappings, misc


def apply_window_policy(image, row, policy):
    if policy == 4:
        image1 = misc.apply_window(image, 40, 80)
        image2 = misc.apply_window(image, 80, 200)
        image3 = misc.apply_window(image, 40, 380)
        image = np.array([image1, image2, image3,]).transpose(1,2,0)
    else:
        raise RuntimeError('Unexpected window policy %s' % policy)
    return image


def apply_dataset_policy(df, policy):
    if policy == 1: # use all records
        pass
    elif policy == 2: # pos == neg
        df_positive = df[df.labels != '']
        df_negative = df[df.labels == '']
        df_sampled = df_negative.sample(len(df_positive))
        df = pd.concat([df_positive, df_sampled], sort=False)
    else:
        raise RuntimeError('Unexpected dataset policy %s' % policy)
    log('applied dataset_policy %s (%d records now)' % (policy, len(df)))
    return df


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, folds):
        self.cfg = cfg

        log(f'dataset_policy: {self.cfg.dataset_policy}')
        log(f'window_policy: {self.cfg.window_policy}')

        self.transforms = factory.get_transforms(self.cfg)
        with open(cfg.annotations, 'rb') as f:
            self.df = pickle.load(f)

        if folds:
            self.df = self.df[self.df.fold.isin(folds)]
            log('read dataset (%d records)' % len(self.df))

        self.df = apply_dataset_policy(self.df, self.cfg.dataset_policy)
        #self.df = self.df.sample(1000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = '%s/%s.dcm' % (self.cfg.imgdir, row.ID)

        dicom = pydicom.dcmread(path)
        image = dicom.pixel_array
        image = misc.rescale_image(image, row.RescaleSlope, row.RescaleIntercept, row.BitsStored, row.PixelRepresentation)
        image = apply_window_policy(image, row, self.cfg.window_policy)

        image = self.transforms(image=image)['image']

        target = np.array([0.0] * len(mappings.label_to_num))
        for label in row.labels.split():
            cls = mappings.label_to_num[label]
            target[cls] = 1.0

        return image, torch.FloatTensor(target), row.ID
