import sys
import os
import argparse
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from tqdm import tqdm

from ..postprocess.make_submission import parse_inputs
from ..utils import mappings
from . import gb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs-oof')
    parser.add_argument('--inputs-test')
    parser.add_argument('--output-dir', default='./meta')
    parser.add_argument('--output-name')
    parser.add_argument('--train-raw', default='./cache/train_raw.pkl')
    parser.add_argument('--test-raw', default='./cache/test_raw.pkl')
    parser.add_argument('--brain-diff', default=60)
    return parser.parse_args()
    

def oof_to_df(predictions):
    records = []
    for i_fold, pred in enumerate(predictions):
        for id,output in zip(pred['ids'], pred['outputs']):
            record = {'ID': id}
            for i in range(6):
                label = mappings.num_to_label[i]
                record.update({
                    label: output[i],
                    'fold': i_fold,
                })
            records.append(record)
    return pd.DataFrame(records)


def test_to_df(pred):
    records = []
    for id,output in zip(pred['ids'], pred['outputs']):
        record = {'ID': id}
        for i in range(6):
            label = mappings.num_to_label[i]
            record.update({
                label: output[i],
            })
        records.append(record)
    return pd.DataFrame(records)


def read_pickled_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def add_gt(df):
    def has_label(labels, label):
        if label in labels:
            return 1
        return 0

    for i in range(6):
        label = mappings.num_to_label[i]
        df['gt_%s' % label] = df.labels.apply(lambda x: has_label(x, label))


def fill_nan_prediction(df):
    for i in range(6):
        label = mappings.num_to_label[i]
        df[label] = df[label].fillna(df[label].min())


def show_stat(df):
    for i in range(6):
        label = mappings.num_to_label[i]
        s = df[label].values
        print('%16s min:%e max:%f mean:%f' % (label, s.min(), s.max(), s.mean()))


def add_position_ord(df):
    expanded = df.ImagePositionPatient.apply(lambda x: pd.Series(x))
    expanded.columns = ['Position1', 'Position2', 'Position3']
    df['Position3'] = expanded.Position3
    df['PositionOrd'] = df.groupby('SeriesInstanceUID')[['Position3']].rank() / df.groupby('SeriesInstanceUID')[['Position3']].transform('count')

    
def create_series_predictions(df):
    def get_or_nan(slices, index):
        try:
            return slices[index]
        except IndexError as e:
            return np.nan

    records = []
    grouped = df.sort_values('PositionOrd').groupby('StudyInstanceUID')
    for _,group in tqdm(grouped, total=len(grouped)):
        rows = [row for row in group.itertuples()]

        for i in range(len(rows)):
            record = {'ID': rows[i].ID}

            for j in range(6):
                label = mappings.num_to_label[j]
                slices = [getattr(row, label) for row in rows]
                ords = [row.PositionOrd for row in rows]

                right = {'%s_r%d' % (label, k):get_or_nan(slices, i+k) for k in range(1, 12)}
                left = {'%s_l%d' % (label, k):get_or_nan(slices, i-k) for k in range(1, 12)}
                record.update({
                    'pos': ords[i],
                    'n_slice': len(rows),
                    'mean_by_study': np.mean(slices),
                })
                record.update(right)
                record.update(left)
            records.append(record)
    return pd.DataFrame(records)


def prepare_train_df(args):

    print('loading train_raw...')
    train_df = read_pickled_data(args.train_raw)

    pred_dfs = []
    df = pd.DataFrame(train_df.ID)
    for inputs in eval(args.inputs_oof):
        oof_predictions = [parse_inputs(elem) for elem in inputs]
        pred_df = oof_to_df(oof_predictions)
        print('raw:%d pred:%d' % (len(df), len(pred_df)))
        pred_df = pd.merge(df, pred_df, on='ID', how='left')
        pred_dfs.append(pred_df)
    print('%d records at total' % len(train_df))

    for i in range(6):
        label = mappings.num_to_label[i]
        train_df[label] = np.nanmean(np.stack([pred_df[label].values for pred_df in pred_dfs]), axis=0)
        train_df.loc[train_df.brain_diff <= args.brain_diff,label] = np.nan
    train_df['fold'] = pred_dfs[0]['fold']

    print('add ground truth to train_df...')
    add_gt(train_df)
    
    print('%d nan predicitons' % train_df['any'].isnull().sum())
    print('fill nan prediction...')
    fill_nan_prediction(train_df)
    print('%d nan predicitons' % train_df['any'].isnull().sum())
    show_stat(train_df)
    
    print('create series predictions...')
    add_position_ord(train_df)
    series_df = create_series_predictions(train_df)
    train_df = pd.merge(train_df, series_df, on='ID')

    return train_df


def prepare_test_df(args):

    print('loading test_raw...')
    test_df = read_pickled_data(args.test_raw)

    pred_dfs = []
    df = pd.DataFrame(test_df.ID)
    for inputs in eval(args.inputs_test):
        test_prediction = parse_inputs(inputs)
        pred_df = test_to_df(test_prediction)
        print('raw:%d pred:%d' % (len(df), len(pred_df)))
        pred_df = pd.merge(df, pred_df, on='ID', how='left')
        pred_dfs.append(pred_df)
    print('%d records at total' % len(test_df))

    for i in range(6):
        label = mappings.num_to_label[i]
        test_df[label] = np.nanmean(np.stack([pred_df[label].values for pred_df in pred_dfs]), axis=0)
        test_df.loc[test_df.brain_diff <= args.brain_diff,label] = np.nan

    print('%d nan predicitons' % test_df['any'].isnull().sum())
    print('fill nan predicition...')
    fill_nan_prediction(test_df)
    print('%d nan predicitons' % test_df['any'].isnull().sum())

    print('create series predictions...')
    add_position_ord(test_df)
    series_df = create_series_predictions(test_df)
    test_df = pd.merge(test_df, series_df, on='ID')

    show_stat(test_df)

    return test_df


def main():
    args = get_args()
    
    pprint(eval(args.inputs_oof))
    pprint(eval(args.inputs_test))

    os.makedirs(args.output_dir, exist_ok=True)

    test_df = prepare_test_df(args)
    train_df = prepare_train_df(args)

    for gb_type in ['xgb', 'lgb', 'cat']:
        oof_all, test_all, logloss_all = gb.run(train_df, test_df, gb_type)

        outputs = [{
            'ids': test_df.ID,
            'oof': oof_all,
            'outputs': np.array(test_all).transpose(1, 0),
            'logloss': logloss_all,
        }]

        path = os.path.join(args.output_dir, '%s_%s.pkl' % (args.output_name, gb_type))
        with open(path, 'wb') as f:
            pickle.dump(outputs, f)
        print('saved %s' % path)


if __name__ == '__main__':
    #print(sys.argv)
    main()
