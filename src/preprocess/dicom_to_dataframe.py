import sys
import argparse
import functools
import pickle
from multiprocessing import Pool
import copy

import pydicom
import pandas as pd
from tqdm import tqdm
import numpy as np
np.seterr(over='ignore')

from ..utils import misc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='provided by kaggle, stage_1_train.csv for stage1')
    parser.add_argument('--output')
    parser.add_argument('--imgdir')
    parser.add_argument('--n-pool', default=6, type=int)
    parser.add_argument('--nrows', default=None, type=int)
    return parser.parse_args()


def group_labels_by_id(df):
    ids = {}
    for row in tqdm(df.itertuples(), total=len(df)):
        prefix, id, label = row.ID.split('_')
        id = '%s_%s' % (prefix, id)
        if id not in ids:
            ids[id] = []
        if row.Label == 1: 
            ids[id].append(label)
    return ids


def remove_corrupted_images(ids):
    ids = ids.copy()
    for id in ['ID_6431af929']:
        try:
            ids.pop(id) 
        except KeyError as e:
            print('%s not found' % id)
        else:
            print('removed %s' % id)

    return ids


def create_record(item, dirname):

    id, labels = item

    path = '%s/%s.dcm' % (dirname, id)
    dicom = pydicom.dcmread(path)
    
    record = {
        'ID': id,
        'labels': ' '.join(labels),
        'n_label': len(labels),
    }
    record.update(misc.get_dicom_raw(dicom))

    raw = dicom.pixel_array
    slope = float(record['RescaleSlope'])
    intercept = float(record['RescaleIntercept'])
    center = misc.get_dicom_value(record['WindowCenter'])
    width = misc.get_dicom_value(record['WindowWidth'])
    bits= record['BitsStored']
    pixel = record['PixelRepresentation']

    image = misc.rescale_image(raw, slope, intercept, bits, pixel)
    doctor = misc.apply_window(image, center, width)
    brain = misc.apply_window(image, 40, 80)

    record.update({
        'raw_max': raw.max(),
        'raw_min': raw.min(),
        'raw_mean': raw.mean(),
        'raw_diff': raw.max() - raw.min(),
        'doctor_max': doctor.max(),
        'doctor_min': doctor.min(),
        'doctor_mean': doctor.mean(),
        'doctor_diff': doctor.max() - doctor.min(),
        'brain_max': brain.max(),
        'brain_min': brain.min(),
        'brain_mean': brain.mean(),
        'brain_diff': brain.max() - brain.min(),
        'brain_ratio': misc.get_windowed_ratio(image, 40, 80),
    })
    return record


def create_df(ids, args):
    print('making records...')
    with Pool(args.n_pool) as pool:
        records = list(tqdm(
            iterable=pool.imap_unordered(
                functools.partial(create_record, dirname=args.imgdir),
                ids.items()
            ),
            total=len(ids),
        ))
    return pd.DataFrame(records).sort_values('ID').reset_index(drop=True)


def main():
    args = get_args()

    df_input = pd.read_csv(args.input, nrows=args.nrows)
    print('read %s (%d records)' % (args.input, len(df_input)))

    ids = group_labels_by_id(df_input)
    ids = remove_corrupted_images(ids)
    
    df_output = create_df(ids, args)

    with open(args.output, 'wb') as f:
        pickle.dump(df_output, f)

    print('converted dicom to dataframe (%d records)' % len(df_output))
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
