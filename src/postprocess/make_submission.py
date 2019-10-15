import sys
import os
import argparse
import pickle
import time

import pandas as pd
import numpy as np

from ..utils import mappings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--inputs', help='for ensembling. can be recursively nested for averaging.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample_submission', default='./input/stage_1_sample_submission.csv')
    parser.add_argument('--clip', type=float, default=1e-6)

    args = parser.parse_args()
    assert args.input or args.inputs

    return args


def avg_predictions(results):
    outputs_all = np.array([result['outputs'] for result in results])
    outputs = outputs_all.mean(axis=0)
    return {
        'ids': results[0]['ids'],
        'outputs': outputs,
    }


def read_prediction(path):
    print('loading %s...' % path)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return avg_predictions(results)
    

def parse_inputs(inputs):
    results = []
    for elem in inputs:
        if type(elem) is list:
            result = parse_inputs(elem)
        else:
            result = read_prediction(elem)
        results.append(result)
    return avg_predictions(results)


def main():
    args = get_args()

    if args.input:
        result = read_prediction(args.input)
    else:
        result = parse_inputs(eval(args.inputs))

    sub = pd.read_csv(args.sample_submission)
    IDs = {}
    for id, outputs in zip(result['ids'], result['outputs']):
        for i, output in enumerate(outputs):
            label = mappings.num_to_label[i]
            ID = '%s_%s' % (id, label)
            IDs[ID] = output

    sub['Label'] = sub.ID.map(IDs)
    sub.loc[sub.Label.isnull(),'Label'] = sub.Label.min()
    if args.clip:
        print('clip values by %e' % args.clip)
        sub['Label'] = np.clip(sub.Label, args.clip, 1-args.clip)

    sub.to_csv(args.output, index=False)
    print(sub.tail())
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
