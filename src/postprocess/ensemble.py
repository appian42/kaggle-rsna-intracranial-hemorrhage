import sys
import os
import argparse
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np

from ..utils import mappings


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--inputs', help='for ensembling. can be recursively nested for averaging.')
    parser.add_argument('--output', required=True)
    # parser.add_argument('--clip', type=float, default=1e-6)

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


def read_prediction(path, dirname=''):
    if dirname:
        path = os.path.join(dirname, path)
    print('loading %s...' % path)
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return avg_predictions(results)
    

def parse_inputs(inputs, dirname=''):
    results = []
    for elem in inputs:
        if type(elem) is list:
            result = parse_inputs(elem, dirname)
        else:
            result = read_prediction(elem, dirname)
        results.append(result)
    return avg_predictions(results)


def main():
    args = get_args()

    if args.input:
        result = read_prediction(args.input)
    else:
        result = parse_inputs(eval(args.inputs))

    preds = defaultdict(list)
    for ID, outputs in zip(result['ids'], result['outputs']):
        preds["ID"].append(ID)
        for i, output in enumerate(outputs):
            label = mappings.num_to_label[i]
            preds[label].append(output)

    preds_df = pd.DataFrame.from_dict(preds) 
    preds_df.to_csv(args.output, index=False)
    print(preds_df.tail())
    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()