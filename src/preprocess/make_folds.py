import sys
import argparse
import collections
import pickle
from pprint import pprint
import random

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--n-fold', type=int, default=5)
    parser.add_argument('--seed', type=int, default=10)
    return parser.parse_args()


def _make_folds(df, n_fold, seed):

    counter_gt = collections.defaultdict(int)
    for labels in df.labels.str.split():
        for label in labels:
            counter_gt[label] += 1

    counter_folds = collections.Counter()

    folds = {}
    min_labels = {}
    random.seed(seed)
    groups = df.groupby('PatientID')
    print('making %d folds...' % n_fold)
    for patient_id, group in tqdm(groups, total=len(groups)):

        labels = []
        for row in group.itertuples():
            for label in row.labels.split():
                labels.append(label)
        if not labels:
            labels = ['']

        count_labels = [counter_gt[label] for label in labels]
        min_label = labels[np.argmin(count_labels)]
        count_folds = [(f, counter_folds[(f, min_label)]) for f in range(n_fold)]
        min_count = min([count for f,count in count_folds])
        fold = random.choice([f for f,count in count_folds if count == min_count])
        folds[patient_id] = fold

        for label in labels:
            counter_folds[(fold,label)] += 1

    pprint(counter_folds)

    return folds


def main():
    args = get_args()
    with open(args.input, 'rb') as f:
        df = pickle.load(f)

    folds = _make_folds(df, args.n_fold, args.seed)
    df['fold'] = df.PatientID.map(folds)
    with open(args.output, 'wb') as f:
        pickle.dump(df, f)

    print('saved to %s' % args.output)


if __name__ == '__main__':
    print(sys.argv)
    main()
